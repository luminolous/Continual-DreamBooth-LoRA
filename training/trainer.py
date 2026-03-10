"""
DreamBooth-LoRA training loop for Stable Diffusion.

Handles:
- Model loading (UNet, text encoder, VAE, tokenizer, scheduler)
- LoRA injection via PEFT (single rolling adapter for legacy, per-task for faithful)
- Per-task token registration and embedding management
- Training loop with Accelerate for mixed precision
- Prior preservation loss support
- Checkpoint save/load for LoRA-only weights + token embeddings
- Composed multi-adapter inference pipeline (shared continual model)
- Memory optimization options (gradient checkpointing, 8-bit Adam, xformers)
- Faithful state save/restore for eval-only and resume
"""

from __future__ import annotations

import copy
import gc
import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config.schema import ModelConfig, TaskConfig, TrainingConfig
from data.dataset import (
    DreamBoothDataset,
    PriorPreservationDataset,
    collate_dreambooth,
)
from utils.io import ensure_dir, save_lora_weights, save_token_embeddings

logger = logging.getLogger(__name__)


class DreamBoothLoRATrainer:
    """Encapsulates DreamBooth-LoRA training for continual learning.

    Supports two modes:
    - Legacy (naive_sequential / c_lora_scaffold): single rolling adapter via inject_lora()
    - Faithful (faithful_c_lora): per-task named adapters via create_task_adapter()
      with composed multi-adapter inference
    """

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
    ):
        self.model_config = model_config
        self.training_config = training_config

        # These are populated by load_models()
        self.unet = None
        self.text_encoder = None
        self.vae = None
        self.tokenizer = None
        self.noise_scheduler = None

        self._models_loaded = False
        self._lora_injected = False
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._weight_dtype = self._get_weight_dtype()

        # Per-task adapter tracking
        self._adapter_names: List[str] = []
        self._task_token_ids: Dict[str, int] = {}  # token_string -> token_id

        # FIX 1: Track gradient hook handles so they can be removed between tasks.
        # This prevents hook accumulation across tasks.
        self._embedding_hook_handle: Optional[torch.utils.hooks.RemovableHook] = None

    def _get_weight_dtype(self) -> torch.dtype:
        """Determine weight dtype from mixed_precision setting."""
        mp = self.model_config.mixed_precision
        if mp == "fp16":
            return torch.float16
        elif mp == "bf16":
            return torch.bfloat16
        return torch.float32

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        """Load the base Stable Diffusion components from HuggingFace."""
        from diffusers import (
            AutoencoderKL,
            DDPMScheduler,
            UNet2DConditionModel,
        )
        from transformers import CLIPTextModel, CLIPTokenizer

        model_id = self.model_config.pretrained_model_name
        logger.info("Loading base model: %s", model_id)

        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=self._weight_dtype
        )
        self.vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", torch_dtype=self._weight_dtype
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=self._weight_dtype
        )
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )

        # Freeze base models
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)

        # Move to device
        self.vae.to(self._device)
        self.text_encoder.to(self._device)
        self.unet.to(self._device)

        # Memory optimizations
        if self.training_config.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        if self.model_config.enable_xformers_memory_efficient_attention:
            try:
                self.unet.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xformers memory efficient attention")
            except Exception as e:
                logger.warning("Could not enable xformers: %s", e)

        self._models_loaded = True
        logger.info("Base models loaded and frozen")

    # ------------------------------------------------------------------
    # Token management (faithful_c_lora)
    # ------------------------------------------------------------------

    def register_task_token(
        self,
        token: str,
        init_mode: str = "random",
    ) -> int:
        """Register a new personalized token for a task.

        Adds the token to the tokenizer and resizes the text encoder's
        embedding layer. In 'random' mode the new embedding row keeps its
        default random initialization from resize.

        Args:
            token: The token string (e.g., "lumy01"). Will be wrapped as
                   "<token>" when added to the tokenizer.
            init_mode: "random" (default init from resize) or "fixed" (no-op).

        Returns:
            The new token's ID in the tokenizer.
        """
        if not self._models_loaded:
            raise RuntimeError("Call load_models() before register_task_token()")

        # Format token with angle brackets for the tokenizer
        token_str = f"<{token}>"

        # Add to tokenizer
        num_added = self.tokenizer.add_tokens([token_str])
        if num_added == 0:
            logger.warning("Token '%s' already in tokenizer vocabulary", token_str)

        # Resize text encoder embeddings to accommodate new token
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        token_id = self.tokenizer.convert_tokens_to_ids(token_str)
        self._task_token_ids[token_str] = token_id

        logger.info(
            "Registered token '%s' -> ID %d (init_mode=%s, vocab_size=%d)",
            token_str, token_id, init_mode, len(self.tokenizer),
        )
        return token_id

    def set_token_embeddings_trainable(self, token_ids: List[int]) -> None:
        """Make only specified token embedding rows trainable.

        FIX 1: This method now properly manages gradient hook lifecycle.
        Any previous hook is explicitly removed before a new one is registered,
        preventing hook accumulation across tasks.

        Freezes the entire text encoder, then unfreezes the embedding layer
        and registers a gradient hook that zeros out gradients for all rows
        except the specified token IDs.

        Args:
            token_ids: List of token IDs whose embeddings should be trained.
        """
        # Remove any previously registered embedding gradient hook
        if self._embedding_hook_handle is not None:
            self._embedding_hook_handle.remove()
            self._embedding_hook_handle = None
            logger.debug("Removed previous embedding gradient hook")

        # Freeze everything in text encoder
        self.text_encoder.requires_grad_(False)

        # Unfreeze embedding layer
        embeddings = self.text_encoder.get_input_embeddings()
        embeddings.weight.requires_grad_(True)

        # Create a gradient mask: zero out gradients for non-target rows.
        # Freeze the target set so the closure captures an immutable copy.
        target_ids = frozenset(token_ids)

        def _embedding_grad_hook(grad):
            mask = torch.zeros_like(grad)
            for tid in target_ids:
                if tid < mask.shape[0]:
                    mask[tid] = 1.0
            return grad * mask

        # Store the handle so we can remove it before the next task
        self._embedding_hook_handle = embeddings.weight.register_hook(
            _embedding_grad_hook
        )

        logger.info(
            "Set token embeddings trainable for %d token(s): %s (hook handle stored)",
            len(token_ids), list(token_ids),
        )

    def clear_token_embedding_hooks(self) -> None:
        """Remove all token embedding gradient hooks.

        Should be called after training is done for a task, or before
        restore/eval to ensure no lingering hooks.
        """
        if self._embedding_hook_handle is not None:
            self._embedding_hook_handle.remove()
            self._embedding_hook_handle = None
            logger.debug("Cleared embedding gradient hook")

    def validate_token_gradients(
        self,
        active_token_ids: List[int],
        all_token_ids: Optional[List[int]] = None,
    ) -> Dict[str, bool]:
        """Smoke test: verify that only active token IDs receive gradients.

        Creates a synthetic forward pass through the embedding layer and
        checks gradient flow. Does NOT modify model state.

        Args:
            active_token_ids: Token IDs that should receive gradients.
            all_token_ids: All task token IDs to check (active + frozen).
                          If None, only checks active tokens.

        Returns:
            Dict with "active_ok" and "frozen_ok" booleans.
        """
        embeddings = self.text_encoder.get_input_embeddings()
        result = {"active_ok": True, "frozen_ok": True}

        # Create a test input containing all relevant tokens
        check_ids = list(active_token_ids)
        if all_token_ids:
            check_ids = list(set(check_ids + all_token_ids))

        # Save current grad state
        had_grad = embeddings.weight.requires_grad

        # Enable grad if needed
        embeddings.weight.requires_grad_(True)

        # Create test input
        test_input = torch.tensor([check_ids], device=embeddings.weight.device)
        test_output = embeddings(test_input)
        test_loss = test_output.sum()
        test_loss.backward()

        if embeddings.weight.grad is not None:
            grad = embeddings.weight.grad
            active_set = set(active_token_ids)

            # Check active tokens have non-zero gradients
            for tid in active_token_ids:
                if tid < grad.shape[0]:
                    if grad[tid].abs().sum().item() == 0:
                        result["active_ok"] = False
                        logger.warning(
                            "GRADIENT CHECK FAILED: active token %d has zero gradient", tid
                        )

            # Check frozen tokens have zero gradients (if hook is working)
            if all_token_ids:
                for tid in all_token_ids:
                    if tid not in active_set and tid < grad.shape[0]:
                        if grad[tid].abs().sum().item() > 0:
                            result["frozen_ok"] = False
                            logger.warning(
                                "GRADIENT CHECK FAILED: frozen token %d has non-zero gradient",
                                tid,
                            )

            # Clean up
            embeddings.weight.grad = None
        else:
            logger.warning("GRADIENT CHECK: no gradient computed on embedding weight")
            result["active_ok"] = False

        # Restore original grad state
        embeddings.weight.requires_grad_(had_grad)

        status = "PASS" if (result["active_ok"] and result["frozen_ok"]) else "FAIL"
        logger.info(
            "Token gradient validation: %s (active_ok=%s, frozen_ok=%s, active=%s)",
            status, result["active_ok"], result["frozen_ok"], active_token_ids,
        )
        return result

    # ------------------------------------------------------------------
    # Per-task adapter lifecycle (faithful_c_lora)
    # ------------------------------------------------------------------

    def create_task_adapter(self, adapter_name: str) -> None:
        """Create a new named LoRA adapter for a task.

        Uses PEFT add_adapter() to create a fresh adapter without
        destroying previous adapters. The new adapter becomes active
        and trainable.

        Args:
            adapter_name: Unique name for this task's adapter (e.g., "task_0").
        """
        from peft import LoraConfig

        if not self._models_loaded:
            raise RuntimeError("Call load_models() before create_task_adapter()")

        lora_config = LoraConfig(
            r=self.model_config.lora_rank,
            lora_alpha=self.model_config.lora_alpha,
            target_modules=self.model_config.lora_target_modules,
            lora_dropout=0.0,
            bias="none",
        )

        if not self._adapter_names:
            # First adapter: wrap with get_peft_model
            from peft import get_peft_model
            self.unet = get_peft_model(self.unet, lora_config, adapter_name=adapter_name)
        else:
            # Subsequent adapters: add_adapter on existing PeftModel
            self.unet.add_adapter(adapter_name, lora_config)

        self.unet.set_adapter(adapter_name)
        self._adapter_names.append(adapter_name)
        self._lora_injected = True

        logger.info(
            "Created adapter '%s' (rank=%d, targets=%s). Total adapters: %d",
            adapter_name,
            self.model_config.lora_rank,
            self.model_config.lora_target_modules,
            len(self._adapter_names),
        )

    def freeze_adapter(self, adapter_name: str) -> None:
        """Freeze a named adapter (set to non-trainable)."""
        from peft import PeftModel
        if isinstance(self.unet, PeftModel):
            for name, param in self.unet.named_parameters():
                if adapter_name in name and "lora" in name.lower():
                    param.requires_grad_(False)
            logger.info("Froze adapter '%s'", adapter_name)

    def get_all_adapter_names(self) -> List[str]:
        """Return list of all registered adapter names."""
        return list(self._adapter_names)

    def get_adapter_lora_factors(
        self,
        adapter_name: str,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Extract raw LoRA factor matrices (A, B) from a named adapter.

        Returns detached CPU copies of each module's lora_A and lora_B weights.

        Args:
            adapter_name: Name of the adapter to extract factors from.

        Returns:
            Dict mapping module_name -> (A_weight, B_weight) where
            A_weight is (rank, in_features) and B_weight is (out_features, rank).
        """
        factors = {}

        for name, param in self.unet.named_parameters():
            if adapter_name not in name:
                continue
            if "lora_A" in name:
                # Extract module base name (everything before .lora_A.adapter_name.weight)
                module_key = name.split(".lora_A.")[0]
                a_weight = param.detach().cpu().clone()

                # Find corresponding B matrix
                b_name = name.replace("lora_A", "lora_B")
                b_param = dict(self.unet.named_parameters()).get(b_name)
                if b_param is not None:
                    b_weight = b_param.detach().cpu().clone()
                    factors[module_key] = (a_weight, b_weight)

        logger.info(
            "Extracted %d LoRA factor pairs from adapter '%s'",
            len(factors), adapter_name,
        )
        return factors

    # ------------------------------------------------------------------
    # Faithful state save/restore (FIX 2)
    # ------------------------------------------------------------------

    def save_faithful_state(self, output_dir: str) -> None:
        """Save the complete faithful_c_lora model state for restore.

        Saves:
        - Tokenizer state (with all added task tokens)
        - All PEFT adapter weights
        - All token embeddings for registered task tokens

        This enables deterministic reconstruction of the shared continual
        model for eval-only and resume.

        Args:
            output_dir: Base experiment output directory.
        """
        state_dir = ensure_dir(Path(output_dir) / "faithful_state")

        # 1. Save tokenizer state (includes all added tokens)
        tokenizer_dir = str(state_dir / "tokenizer")
        self.tokenizer.save_pretrained(tokenizer_dir)
        logger.info("Saved tokenizer state to %s", tokenizer_dir)

        # 2. Save all PEFT adapter weights
        from peft import PeftModel
        if isinstance(self.unet, PeftModel):
            adapters_dir = str(state_dir / "adapters")
            self.unet.save_pretrained(adapters_dir)
            logger.info("Saved all PEFT adapters to %s", adapters_dir)

        # 3. Save all task token embeddings
        if self._task_token_ids:
            all_token_ids = list(self._task_token_ids.values())
            emb_path = str(state_dir / "all_token_embeddings.pt")
            save_token_embeddings(self.text_encoder, all_token_ids, emb_path)

        # 4. Save adapter name list for restore ordering
        from utils.io import save_json
        save_json({
            "adapter_names": self._adapter_names,
            "task_token_ids": {k: v for k, v in self._task_token_ids.items()},
        }, str(state_dir / "restore_metadata.json"))

        logger.info(
            "Faithful state saved: %d adapters, %d tokens -> %s",
            len(self._adapter_names), len(self._task_token_ids), state_dir,
        )

    def restore_faithful_state(
        self,
        output_dir: str,
        adapter_names_to_load: Optional[List[str]] = None,
    ) -> None:
        """Restore the complete faithful_c_lora model state from checkpoint.

        Reconstructs:
        1. Tokenizer with all added task tokens
        2. Text encoder with resized embeddings
        3. All or selected PEFT adapters
        4. Learned token embeddings

        After this call, the trainer is ready for inference with the
        shared continual model (all adapters composed).

        Args:
            output_dir: Base experiment output directory.
            adapter_names_to_load: Optional subset of adapters to load.
                                  If None, loads all adapters from checkpoint.
        """
        if not self._models_loaded:
            raise RuntimeError("Call load_models() before restore_faithful_state()")

        state_dir = Path(output_dir) / "faithful_state"
        if not state_dir.is_dir():
            raise FileNotFoundError(
                f"Faithful state directory not found: {state_dir}. "
                "Was the experiment run with faithful_c_lora method?"
            )

        from utils.io import load_json, load_token_embeddings

        # 1. Load restore metadata
        meta_path = state_dir / "restore_metadata.json"
        if meta_path.exists():
            meta = load_json(str(meta_path))
            saved_adapter_names = meta.get("adapter_names", [])
            saved_token_ids = meta.get("task_token_ids", {})
        else:
            saved_adapter_names = []
            saved_token_ids = {}

        # 2. Restore tokenizer (includes all added tokens)
        tokenizer_dir = state_dir / "tokenizer"
        if tokenizer_dir.is_dir():
            from transformers import CLIPTokenizer
            self.tokenizer = CLIPTokenizer.from_pretrained(str(tokenizer_dir))
            # Resize text encoder to match tokenizer
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))
            logger.info(
                "Restored tokenizer from %s (vocab_size=%d)",
                tokenizer_dir, len(self.tokenizer),
            )
        else:
            logger.warning("No saved tokenizer found at %s", tokenizer_dir)

        # 3. Restore PEFT adapters
        adapters_dir = state_dir / "adapters"
        if adapters_dir.is_dir():
            from peft import PeftModel
            names_to_load = adapter_names_to_load or saved_adapter_names

            if names_to_load:
                # Load first adapter
                first_name = names_to_load[0]
                self.unet = PeftModel.from_pretrained(
                    self.unet,
                    str(adapters_dir),
                    adapter_name=first_name,
                    is_trainable=False,
                )
                self._adapter_names = [first_name]
                self._lora_injected = True

                # Load remaining adapters
                for name in names_to_load[1:]:
                    try:
                        self.unet.load_adapter(
                            str(adapters_dir),
                            adapter_name=name,
                            is_trainable=False,
                        )
                        self._adapter_names.append(name)
                    except Exception as e:
                        logger.warning("Could not load adapter '%s': %s", name, e)

                logger.info(
                    "Restored %d adapters: %s",
                    len(self._adapter_names), self._adapter_names,
                )
        else:
            logger.warning("No saved adapters found at %s", adapters_dir)

        # 4. Restore token embeddings
        emb_path = state_dir / "all_token_embeddings.pt"
        if emb_path.exists():
            load_token_embeddings(self.text_encoder, self.tokenizer, str(emb_path))

        # 5. Restore token ID mapping
        self._task_token_ids = {k: int(v) for k, v in saved_token_ids.items()}

        # 6. Clear any lingering gradient hooks (not needed for inference)
        self.clear_token_embedding_hooks()

        logger.info(
            "Faithful state restored: %d adapters, %d tokens, ready for inference",
            len(self._adapter_names), len(self._task_token_ids),
        )

    # ------------------------------------------------------------------
    # Legacy single-adapter methods (naive_sequential / c_lora_scaffold)
    # ------------------------------------------------------------------

    def inject_lora(self, prev_checkpoint: Optional[str] = None) -> None:
        """Inject LoRA adapters into the UNet (legacy single-adapter mode).

        If prev_checkpoint is provided, load the existing LoRA weights first
        (for continual learning), then ensure they're trainable.

        Args:
            prev_checkpoint: Path to directory with previous LoRA weights.
        """
        from peft import LoraConfig, get_peft_model

        if not self._models_loaded:
            raise RuntimeError("Call load_models() before inject_lora()")

        if prev_checkpoint is not None:
            logger.info("Loading previous LoRA checkpoint from %s", prev_checkpoint)
            from peft import PeftModel

            if self._lora_injected:
                self.unet = self.unet.merge_and_unload()
                self._lora_injected = False

            self.unet = PeftModel.from_pretrained(
                self.unet,
                prev_checkpoint,
                is_trainable=True,
            )
            self._lora_injected = True
            logger.info("Loaded and set previous LoRA as trainable")
        else:
            lora_config = LoraConfig(
                r=self.model_config.lora_rank,
                lora_alpha=self.model_config.lora_alpha,
                target_modules=self.model_config.lora_target_modules,
                lora_dropout=0.0,
                bias="none",
            )
            self.unet = get_peft_model(self.unet, lora_config)
            self._lora_injected = True
            logger.info(
                "Injected fresh LoRA (rank=%d, alpha=%d, targets=%s)",
                self.model_config.lora_rank,
                self.model_config.lora_alpha,
                self.model_config.lora_target_modules,
            )

        trainable = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.unet.parameters())
        logger.info(
            "Trainable params: %d / %d (%.2f%%)",
            trainable, total, 100.0 * trainable / total,
        )

    def get_lora_params(self) -> Dict[str, torch.nn.Parameter]:
        """Return a dict of trainable LoRA parameters (for regularization)."""
        return {
            name: param
            for name, param in self.unet.named_parameters()
            if param.requires_grad and "lora" in name.lower()
        }

    def snapshot_lora_weights(self) -> Dict[str, torch.Tensor]:
        """Take a detached CPU snapshot of current LoRA weights."""
        return {
            name: param.detach().cpu().clone()
            for name, param in self.get_lora_params().items()
        }

    # ------------------------------------------------------------------
    # Prior preservation image generation
    # ------------------------------------------------------------------

    def generate_class_prior_images(
        self,
        class_prompt: str,
        num_images: int,
        output_dir: str,
        batch_size: int = 4,
    ) -> str:
        """Generate class-prior images using the base model (no LoRA).

        Used for prior preservation regularization. Images are generated
        once before the first task and reused for all subsequent tasks.

        Args:
            class_prompt: The class prompt (e.g., "a photo of anime character").
            num_images: Number of images to generate.
            output_dir: Directory to save images.
            batch_size: Generation batch size.

        Returns:
            Path to the class images directory.
        """
        from diffusers import StableDiffusionPipeline

        out_path = ensure_dir(output_dir)

        # Check if already generated
        from data.dataset import find_images
        existing = find_images(out_path)
        if len(existing) >= num_images:
            logger.info(
                "Found %d existing class prior images in %s, skipping generation",
                len(existing), out_path,
            )
            return str(out_path)

        logger.info(
            "Generating %d class prior images with prompt '%s'",
            num_images, class_prompt,
        )

        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_config.pretrained_model_name,
            torch_dtype=self._weight_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe = pipe.to(self._device)
        pipe.set_progress_bar_config(disable=True)

        generated = 0
        while generated < num_images:
            current_batch = min(batch_size, num_images - generated)
            generator = torch.Generator(device=self._device).manual_seed(
                42 + generated
            )
            with torch.no_grad():
                images = pipe(
                    prompt=[class_prompt] * current_batch,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    generator=generator,
                ).images

            for img in images:
                img.save(str(out_path / f"class_{generated:04d}.png"))
                generated += 1

        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Generated %d class prior images in %s", generated, out_path)
        return str(out_path)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        task: TaskConfig,
        output_dir: str,
        extra_loss_fn: Optional[Callable] = None,
        prompt_override: Optional[str] = None,
        prior_dataset: Optional[Any] = None,
        prior_loss_weight: float = 1.0,
        token_ids_to_save: Optional[List[int]] = None,
    ) -> str:
        """Run DreamBooth-LoRA training for a single task.

        Args:
            task: Task configuration with prompt and data paths.
            output_dir: Where to save the LoRA checkpoint after training.
            extra_loss_fn: Optional callable(unet) -> loss_tensor for method
                           regularization (e.g., occupancy constraint).
            prompt_override: Optional prompt string to use instead of task.instance_prompt.
            prior_dataset: Optional PriorPreservationDataset for prior preservation loss.
            prior_loss_weight: Weight for prior preservation loss term.
            token_ids_to_save: Optional list of token IDs whose embeddings to save.

        Returns:
            Path to the saved LoRA checkpoint directory.
        """
        if not self._lora_injected:
            raise RuntimeError("Call inject_lora() or create_task_adapter() before train()")

        tc = self.training_config
        instance_prompt = prompt_override or task.instance_prompt

        # Build instance dataset and dataloader
        dataset = DreamBoothDataset(
            data_dir=task.data_dir,
            instance_prompt=instance_prompt,
            tokenizer=self.tokenizer,
            size=512,
            center_crop=True,
            repeats=max(1, tc.max_train_steps // 10),
        )

        dataloader = DataLoader(
            dataset,
            batch_size=tc.train_batch_size,
            shuffle=True,
            collate_fn=collate_dreambooth,
            num_workers=0,
            pin_memory=True,
        )

        # Prior preservation dataloader (optional)
        prior_dataloader = None
        prior_iter = None
        if prior_dataset is not None:
            prior_dataloader = DataLoader(
                prior_dataset,
                batch_size=tc.train_batch_size,
                shuffle=True,
                collate_fn=collate_dreambooth,
                num_workers=0,
                pin_memory=True,
            )
            prior_iter = iter(prior_dataloader)

        # Collect trainable parameters: UNet LoRA + optionally token embeddings
        trainable_params = [p for p in self.unet.parameters() if p.requires_grad]

        # Include text encoder embedding params if any are trainable
        for p in self.text_encoder.parameters():
            if p.requires_grad:
                trainable_params.append(p)

        if tc.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW8bit(
                    trainable_params, lr=tc.learning_rate
                )
                logger.info("Using 8-bit AdamW optimizer")
            except ImportError:
                logger.warning(
                    "bitsandbytes not available, falling back to standard AdamW"
                )
                optimizer = torch.optim.AdamW(trainable_params, lr=tc.learning_rate)
        else:
            optimizer = torch.optim.AdamW(trainable_params, lr=tc.learning_rate)

        # Training loop
        self.unet.train()
        global_step = 0
        data_iter = iter(dataloader)

        progress = tqdm(
            range(tc.max_train_steps),
            desc=f"Training [{task.name}]",
            unit="step",
        )

        for step in progress:
            # Get instance batch (cycle if exhausted)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            pixel_values = batch["pixel_values"].to(
                self._device, dtype=self._weight_dtype
            )
            input_ids = batch["input_ids"].to(self._device)

            # Encode images to latent space
            with torch.no_grad():
                latents = self.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor

            # Sample noise and timesteps
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (bsz,), device=self._device, dtype=torch.long,
            )

            # Add noise to latents
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            # Get text embeddings (with gradient if training token embeddings)
            if any(p.requires_grad for p in self.text_encoder.parameters()):
                encoder_hidden_states = self.text_encoder(input_ids)[0]
            else:
                with torch.no_grad():
                    encoder_hidden_states = self.text_encoder(input_ids)[0]

            if encoder_hidden_states.dtype != self._weight_dtype:
                encoder_hidden_states = encoder_hidden_states.to(self._weight_dtype)

            # Predict noise (instance)
            model_pred = self.unet(
                noisy_latents, timesteps, encoder_hidden_states
            ).sample

            # Instance loss
            instance_loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            loss = instance_loss

            # Prior preservation loss (if enabled)
            if prior_iter is not None:
                try:
                    prior_batch = next(prior_iter)
                except StopIteration:
                    prior_iter = iter(prior_dataloader)
                    prior_batch = next(prior_iter)

                prior_pixels = prior_batch["pixel_values"].to(
                    self._device, dtype=self._weight_dtype
                )
                prior_ids = prior_batch["input_ids"].to(self._device)

                with torch.no_grad():
                    prior_latents = self.vae.encode(prior_pixels).latent_dist.sample()
                    prior_latents = prior_latents * self.vae.config.scaling_factor

                prior_noise = torch.randn_like(prior_latents)
                prior_timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps,
                    (prior_latents.shape[0],), device=self._device, dtype=torch.long,
                )
                prior_noisy = self.noise_scheduler.add_noise(
                    prior_latents, prior_noise, prior_timesteps
                )

                with torch.no_grad():
                    prior_enc = self.text_encoder(prior_ids)[0]
                    if prior_enc.dtype != self._weight_dtype:
                        prior_enc = prior_enc.to(self._weight_dtype)

                prior_pred = self.unet(
                    prior_noisy, prior_timesteps, prior_enc
                ).sample
                prior_loss = F.mse_loss(
                    prior_pred.float(), prior_noise.float(), reduction="mean"
                )
                loss = loss + prior_loss_weight * prior_loss

            # Add regularization from continual method (if any)
            if extra_loss_fn is not None:
                reg_loss = extra_loss_fn(self.unet)
                loss = loss + reg_loss

            # Backward + optimize
            loss.backward()

            if (step + 1) % tc.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        progress.close()
        logger.info(
            "Training complete for '%s': %d steps, final loss=%.4f",
            task.name, global_step, loss.item(),
        )

        # Save LoRA checkpoint
        ckpt_dir = str(ensure_dir(Path(output_dir) / "lora_weights"))
        save_lora_weights(self.unet, ckpt_dir)
        logger.info("Checkpoint saved to %s", ckpt_dir)

        # Save token embeddings if specified
        if token_ids_to_save:
            token_emb_path = str(Path(output_dir) / "token_embeddings.pt")
            save_token_embeddings(self.text_encoder, token_ids_to_save, token_emb_path)

        return ckpt_dir

    # ------------------------------------------------------------------
    # Inference pipeline construction
    # ------------------------------------------------------------------

    def build_inference_pipeline(
        self,
        adapter_names: Optional[List[str]] = None,
        token_embeddings_dir: Optional[str] = None,
        lora_dir: Optional[str] = None,
    ):
        """Build a StableDiffusionPipeline with the shared continual model.

        For faithful_c_lora: uses the trainer's UNet (which holds all adapters)
        and text encoder (which holds all token embeddings). No heavy model
        reload — just references the in-memory models.

        For legacy: loads a single LoRA checkpoint into a fresh pipeline.

        Args:
            adapter_names: List of adapter names to activate (faithful mode).
            token_embeddings_dir: Legacy: directory with token_embeddings.pt files.
            lora_dir: Legacy: path to single LoRA checkpoint directory.

        Returns:
            A StableDiffusionPipeline ready for generation.
        """
        from diffusers import StableDiffusionPipeline

        if adapter_names and self._adapter_names:
            # Faithful mode: build pipeline from in-memory models (no reload)
            pipe = StableDiffusionPipeline.from_pretrained(
                self.model_config.pretrained_model_name,
                unet=self.unet,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                torch_dtype=self._weight_dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
            pipe = pipe.to(self._device)

            # Set all specified adapters active for composed inference
            from peft import PeftModel
            if isinstance(self.unet, PeftModel):
                try:
                    self.unet.set_adapter(adapter_names)
                except Exception:
                    for name in adapter_names:
                        try:
                            self.unet.set_adapter(name)
                        except Exception as e:
                            logger.warning("Could not set adapter '%s': %s", name, e)

            logger.info(
                "Built composed inference pipeline with %d adapters: %s",
                len(adapter_names), adapter_names,
            )

        else:
            # Legacy mode: fresh pipeline + single LoRA
            pipe = StableDiffusionPipeline.from_pretrained(
                self.model_config.pretrained_model_name,
                torch_dtype=self._weight_dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
            pipe = pipe.to(self._device)

            if lora_dir is not None:
                from utils.io import load_lora_weights_into_pipeline
                load_lora_weights_into_pipeline(pipe, lora_dir)

            # Load token embeddings if directory provided
            if token_embeddings_dir:
                from utils.io import load_token_embeddings
                emb_dir = Path(token_embeddings_dir)
                for emb_file in sorted(emb_dir.rglob("token_embeddings.pt")):
                    try:
                        pipe.tokenizer = self.tokenizer
                        pipe.text_encoder = self.text_encoder
                        load_token_embeddings(pipe.text_encoder, pipe.tokenizer, emb_file)
                    except Exception as e:
                        logger.warning("Could not load embeddings from %s: %s", emb_file, e)

        if self.model_config.enable_xformers_memory_efficient_attention:
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

        return pipe

    def build_diagnostic_pipeline(
        self,
        adapter_name: str,
        token_embedding_path: Optional[str] = None,
    ):
        """Build a pipeline with a single isolated adapter (diagnostic mode).

        This loads ONLY the specified adapter for isolated evaluation.
        Used for debugging and ablation, NOT for primary evaluation.

        Args:
            adapter_name: Single adapter to activate.
            token_embedding_path: Path to that task's token_embeddings.pt.

        Returns:
            A StableDiffusionPipeline with single adapter active.
        """
        from diffusers import StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_config.pretrained_model_name,
            unet=self.unet,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            torch_dtype=self._weight_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe = pipe.to(self._device)

        # Activate only the specified adapter
        from peft import PeftModel
        if isinstance(self.unet, PeftModel):
            self.unet.set_adapter(adapter_name)

        if token_embedding_path:
            from utils.io import load_token_embeddings
            load_token_embeddings(pipe.text_encoder, pipe.tokenizer, token_embedding_path)

        logger.info("Built diagnostic pipeline with adapter '%s' only", adapter_name)
        return pipe
