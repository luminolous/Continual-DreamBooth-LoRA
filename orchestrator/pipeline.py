"""
Continual learning pipeline orchestrator.

Controls the sequential task loop:
1. For each task: setup adapters → train → archive → evaluate all seen tasks
2. Build the score matrix incrementally
3. Generate final report after all tasks complete

Supports three methods:
- naive_sequential / c_lora_scaffold: legacy single-adapter flow
- faithful_c_lora: shared continual model with per-task adapters, token
  embeddings, occupancy regularization, and composed-model evaluation

Supports eval-only and resume modes for faithful_c_lora:
- eval-only: restore full continual state from checkpoint, run composed eval
- resume: restore state, continue training from next uncompleted task
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from config.schema import PipelineConfig
from data.dataset import (
    PriorPreservationDataset,
    build_task_prompt,
    validate_task_data,
)
from eval.generator import generate_eval_images
from eval.metrics import (
    build_score_matrix,
    compute_ccip_score,
    compute_confusion_gap,
    compute_forgetting_metrics,
    compute_per_prompt_scores,
)
from eval.report import save_full_report
from methods.base import BaseMethod
from training.trainer import DreamBoothLoRATrainer
from utils.io import (
    ensure_dir,
    load_task_registry,
    save_json,
    save_task_info,
    save_task_registry,
)

logger = logging.getLogger(__name__)


def create_method(config: PipelineConfig) -> BaseMethod:
    """Instantiate the continual learning method from config.

    Args:
        config: Pipeline configuration.

    Returns:
        A BaseMethod subclass instance.

    Raises:
        ValueError: If the method name is unknown.
    """
    method_name = config.experiment.method

    if method_name == "naive_sequential":
        from methods.naive_sequential import NaiveSequential
        return NaiveSequential()

    elif method_name == "c_lora_scaffold":
        from methods.c_lora_scaffold import CLoRAScaffold
        return CLoRAScaffold(
            regularization_weight=config.c_lora.regularization_weight,
            importance_method=config.c_lora.importance_method,
            importance_decay=config.c_lora.importance_decay,
        )

    elif method_name == "faithful_c_lora":
        from methods.faithful_c_lora import FaithfulCLoRA
        return FaithfulCLoRA(
            regularization_weight=config.c_lora.regularization_weight,
            token_init=config.c_lora.token_init,
        )

    else:
        raise ValueError(f"Unknown method: {method_name}")


class ContinualPipeline:
    """Orchestrates the continual DreamBooth-LoRA training and evaluation loop.

    For faithful_c_lora:
    - Primary evaluation uses the composed model (all adapters active)
    - Task identity comes from personalized token embeddings
    - Forgetting is measured in the composed model, not by loading isolated adapters

    Usage:
        pipeline = ContinualPipeline(config)
        pipeline.run()
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.method = create_method(config)
        self.trainer = DreamBoothLoRATrainer(
            model_config=config.model,
            training_config=config.training,
        )

        # Score tracking: scores[stage_t][task_j] = ccip_score
        self.scores: Dict[int, Dict[int, float]] = {}

        # Diagnostic scores (isolated adapter eval): diag_scores[task_j] = score
        self.diagnostic_scores: Dict[int, float] = {}

        # Phase 2 data
        self.confusion_data: Dict[int, Dict[int, dict]] = {}
        self.per_prompt_data: Dict[int, Dict[int, dict]] = {}

        # Task metadata registry
        self.task_registry: Dict = {
            "method": config.experiment.method,
            "base_model": config.model.pretrained_model_name,
            "seed": config.experiment.seed,
            "lora_rank": config.model.lora_rank,
            "lora_alpha": config.model.lora_alpha,
            "lora_target_modules": config.model.lora_target_modules,
            "num_tasks_completed": 0,
            "composition_policy": "compose_all",
            "tokenizer_snapshot": "faithful_state/tokenizer",
            "adapter_snapshot": "faithful_state/adapters",
            "token_embedding_snapshot": "faithful_state/all_token_embeddings.pt",
            "tasks": [],
        }

        # Token IDs per task index
        self._task_token_ids: Dict[int, int] = {}

        # Output directories
        self.base_output = ensure_dir(
            Path(config.experiment.output_dir) / config.experiment.name
        )
        self.checkpoints_dir = ensure_dir(self.base_output / "checkpoints")
        self.eval_dir = ensure_dir(self.base_output / "eval_images")

    def validate_all_tasks(self) -> None:
        """Validate all task datasets before starting training."""
        logger.info("Validating %d task datasets...", len(self.config.tasks))
        for task in self.config.tasks:
            validate_task_data(task)
        logger.info("All task datasets validated successfully")

    # ------------------------------------------------------------------
    # Faithful C-LoRA flow
    # ------------------------------------------------------------------

    def _run_faithful(self) -> None:
        """Run the faithful C-LoRA pipeline.

        Uses shared continual model semantics:
        - Per-task adapters composed into single effective delta
        - Task identity from token embeddings
        - Occupancy regularization
        - Composed-model evaluation (primary)
        """
        tasks = self.config.tasks
        num_tasks = len(tasks)
        c_lora_cfg = self.config.c_lora

        logger.info("=" * 60)
        logger.info("FAITHFUL C-LORA PIPELINE")
        logger.info("Tasks: %d", num_tasks)
        logger.info("Regularizer: %s (λ=%.4f)", c_lora_cfg.regularizer_type,
                     c_lora_cfg.regularization_weight)
        logger.info("Token init: %s", c_lora_cfg.token_init)
        logger.info("Prompt mode: %s", c_lora_cfg.prompt_mode)
        logger.info("Prior preservation: %s", c_lora_cfg.prior_preservation)
        logger.info("Output: %s", self.base_output)
        logger.info("=" * 60)

        self.validate_all_tasks()
        self.trainer.load_models()

        # Generate class prior images if prior preservation is enabled
        prior_dataset = None
        if c_lora_cfg.prior_preservation:
            class_images_dir = self.trainer.generate_class_prior_images(
                class_prompt=c_lora_cfg.class_prompt,
                num_images=c_lora_cfg.num_class_images,
                output_dir=str(self.base_output / "class_prior"),
            )
            prior_dataset = PriorPreservationDataset(
                class_images_dir=class_images_dir,
                class_prompt=c_lora_cfg.class_prompt,
                tokenizer=self.trainer.tokenizer,
                size=512,
            )

        for t, task in enumerate(tasks):
            stage_start = time.time()
            logger.info("")
            logger.info("=" * 60)
            logger.info("TASK %d/%d: %s", t + 1, num_tasks, task.name)
            logger.info("=" * 60)

            # 1. Pre-task: archive past factors, activate occupancy
            self.method.pre_task_setup(t, self.trainer, self.config)

            # 2. Setup adapters: create adapter, register token, freeze old
            self.method.setup_adapters(t, self.trainer, self.config)

            # 3. Build prompt for this task
            prompt = build_task_prompt(task, c_lora_cfg.prompt_mode)
            logger.info("Training prompt: '%s'", prompt)

            # 4. Get regularization loss function
            extra_loss_fn = self.method.get_extra_loss_fn()

            # 5. Validate token gradients (smoke test)
            token_id = self.trainer._task_token_ids.get(f"<{task.trigger_token}>")
            all_token_ids = list(self.trainer._task_token_ids.values())
            if token_id is not None and c_lora_cfg.train_token_embeddings:
                self.trainer.validate_token_gradients(
                    active_token_ids=[token_id],
                    all_token_ids=all_token_ids,
                )

            # 6. Train
            task_output_dir = str(
                self.checkpoints_dir / f"task_{t:02d}_{task.name}"
            )
            token_ids_to_save = [token_id] if token_id is not None else None

            self.trainer.train(
                task=task,
                output_dir=task_output_dir,
                extra_loss_fn=extra_loss_fn,
                prompt_override=prompt,
                prior_dataset=prior_dataset,
                prior_loss_weight=c_lora_cfg.prior_loss_weight,
                token_ids_to_save=token_ids_to_save,
            )

            # 7. Post-task: archive factors, update occupancy, clear hooks
            self.method.post_task_cleanup(t, self.trainer)
            self.trainer.clear_token_embedding_hooks()

            # 8. Save task metadata (strengthened for restore)
            task_info = {
                "task_name": task.name,
                "task_idx": t,
                "task_order": t,
                "adapter_name": f"task_{t}",
                "token": f"<{task.trigger_token}>",
                "token_id": token_id,
                "prompt_mode": c_lora_cfg.prompt_mode,
                "instance_prompt": prompt,
                "class_prompt": c_lora_cfg.class_prompt,
                "regularizer_type": c_lora_cfg.regularizer_type,
                "regularization_weight": c_lora_cfg.regularization_weight,
                "lora_rank": self.config.model.lora_rank,
                "lora_alpha": self.config.model.lora_alpha,
                "lora_target_modules": self.config.model.lora_target_modules,
                "max_train_steps": self.config.training.max_train_steps,
                "seed": self.config.experiment.seed,
                "checkpoint_dir": task_output_dir,
                "token_embedding_path": str(Path(task_output_dir) / "token_embeddings.pt"),
                "adapter_weights_path": str(Path(task_output_dir) / "lora_weights"),
            }
            save_task_info(task_output_dir, task_info)
            self.task_registry["tasks"].append(task_info)
            self.task_registry["num_tasks_completed"] = t + 1
            save_task_registry(str(self.base_output), self.task_registry)

            # 9. Save full faithful state (for restore / eval-only / resume)
            self.trainer.save_faithful_state(str(self.base_output))

            # 10. PRIMARY EVAL: composed model (all adapters active)
            logger.info(
                "PRIMARY EVAL: composed model after task %d (adapters: %s)",
                t, self.trainer.get_all_adapter_names(),
            )
            pipe = self.trainer.build_inference_pipeline(
                adapter_names=self.trainer.get_all_adapter_names(),
            )
            self._evaluate_stage(t, pipe)

            del pipe
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 11. SECONDARY: diagnostic eval (optional)
            if c_lora_cfg.run_diagnostic_eval:
                self._run_diagnostic_eval(t)

            # 12. Multi-concept probe (optional, if >= 2 tasks)
            if c_lora_cfg.run_multi_concept_probe and t >= 1:
                self._run_multi_concept_probe(t)

            # Save intermediate scores
            save_json(self.scores, str(self.base_output / "scores_intermediate.json"))

            stage_time = time.time() - stage_start
            logger.info("Task %d complete in %.1f seconds", t, stage_time)

        # Final report
        self._generate_report(num_tasks)

    def _run_diagnostic_eval(self, stage_t: int) -> None:
        """Run isolated-adapter diagnostic evaluation.

        Loads each task's adapter in isolation to verify adapter integrity.
        This is NOT the primary metric — it's a debugging/ablation tool.
        """
        tasks = self.config.tasks
        logger.info("DIAGNOSTIC EVAL: isolated adapters after task %d", stage_t)

        for j in range(stage_t + 1):
            task = tasks[j]
            adapter_name = f"task_{j}"
            task_dir = self.checkpoints_dir / f"task_{j:02d}_{task.name}"
            emb_path = str(task_dir / "token_embeddings.pt")

            pipe = self.trainer.build_diagnostic_pipeline(
                adapter_name=adapter_name,
                token_embedding_path=emb_path if Path(emb_path).exists() else None,
            )

            prompt = build_task_prompt(task, self.config.c_lora.prompt_mode)
            eval_output_dir = str(
                self.eval_dir / f"diagnostic" / f"task_{j:02d}_{task.name}"
            )

            generate_eval_images(
                pipeline=pipe,
                task=task,
                eval_config=self.config.evaluation,
                output_dir=eval_output_dir,
                seed=self.config.experiment.seed,
                prompt_mode=self.config.c_lora.prompt_mode,
            )

            score = compute_ccip_score(
                generated_dir=eval_output_dir,
                ref_dir=task.ref_dir,
            )
            self.diagnostic_scores[j] = score
            logger.info(
                "  DIAGNOSTIC[task=%s, isolated] = %.4f", task.name, score
            )

            # Restore all adapters active after diagnostic
            from peft import PeftModel
            if isinstance(self.trainer.unet, PeftModel):
                self.trainer.unet.set_adapter(self.trainer.get_all_adapter_names())

            del pipe
            gc.collect()

        # Save diagnostic scores
        save_json(
            self.diagnostic_scores,
            str(self.base_output / "diagnostic_scores.json"),
        )

    def _run_multi_concept_probe(self, stage_t: int) -> None:
        """Generate multi-concept images with multiple task tokens.

        Lightweight visual probe — generates pair-wise multi-token images.
        No automated metric in MVP; for visual inspection only.
        """
        tasks = self.config.tasks
        logger.info("MULTI-CONCEPT PROBE after task %d", stage_t)

        pipe = self.trainer.build_inference_pipeline(
            adapter_names=self.trainer.get_all_adapter_names(),
        )
        pipe.set_progress_bar_config(disable=True)

        mc_dir = ensure_dir(self.eval_dir / f"stage_{stage_t:02d}" / "multi_concept")

        for j in range(stage_t):
            for k in range(j + 1, stage_t + 1):
                task_j = tasks[j]
                task_k = tasks[k]
                token_j = f"<{task_j.trigger_token}>"
                token_k = f"<{task_k.trigger_token}>"

                prompt = f"a photo of {token_j} and {token_k}"
                logger.info("  Multi-concept: '%s'", prompt)

                for img_i in range(2):
                    generator = torch.Generator(device=pipe.device).manual_seed(
                        self.config.experiment.seed + img_i
                    )
                    with torch.no_grad():
                        result = pipe(
                            prompt=prompt,
                            num_inference_steps=self.config.evaluation.num_inference_steps,
                            guidance_scale=self.config.evaluation.guidance_scale,
                            generator=generator,
                        )
                    fname = f"{task_j.name}+{task_k.name}_i{img_i:02d}.png"
                    result.images[0].save(str(mc_dir / fname))

        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Evaluation helpers (shared)
    # ------------------------------------------------------------------

    def _evaluate_stage(self, stage_t: int, pipe) -> None:
        """Evaluate all seen tasks after training stage t.

        For faithful_c_lora: pipe is the composed model with all adapters active.
        For legacy methods: pipe is the single-adapter pipeline.
        """
        tasks = self.config.tasks
        eval_config = self.config.evaluation
        c_lora_cfg = self.config.c_lora

        self.scores[stage_t] = {}
        if eval_config.compute_confusion_gap:
            self.confusion_data[stage_t] = {}
        if eval_config.per_prompt_breakdown:
            self.per_prompt_data[stage_t] = {}

        # Determine prompt_mode (only relevant for faithful_c_lora)
        prompt_mode = None
        if self.config.experiment.method == "faithful_c_lora":
            prompt_mode = c_lora_cfg.prompt_mode

        for j in range(stage_t + 1):
            eval_task = tasks[j]
            eval_output_dir = str(
                self.eval_dir / f"stage_{stage_t:02d}" / eval_task.name
            )

            generate_eval_images(
                pipeline=pipe,
                task=eval_task,
                eval_config=eval_config,
                output_dir=eval_output_dir,
                seed=self.config.experiment.seed,
                prompt_mode=prompt_mode,
            )

            score = compute_ccip_score(
                generated_dir=eval_output_dir,
                ref_dir=eval_task.ref_dir,
            )
            self.scores[stage_t][j] = score
            logger.info(
                "  CCIP[stage=%d, task=%s] = %.4f", stage_t, eval_task.name, score
            )

            # Per-prompt breakdown
            if eval_config.per_prompt_breakdown:
                prompt_scores = compute_per_prompt_scores(
                    generated_dir=eval_output_dir,
                    ref_dir=eval_task.ref_dir,
                    num_prompts=len(eval_config.prompts_per_character),
                    num_images_per_prompt=eval_config.num_images_per_prompt,
                    task_name=eval_task.name,
                )
                self.per_prompt_data[stage_t][j] = prompt_scores

            # Confusion gap
            if eval_config.compute_confusion_gap and stage_t > 0:
                other_refs = [
                    (tasks[k].name, tasks[k].ref_dir)
                    for k in range(stage_t + 1) if k != j
                ]
                if other_refs:
                    gap_data = compute_confusion_gap(
                        generated_dir=eval_output_dir,
                        target_ref_dir=eval_task.ref_dir,
                        other_ref_dirs=other_refs,
                    )
                    self.confusion_data[stage_t][j] = gap_data

    # ------------------------------------------------------------------
    # Legacy flow (naive_sequential / c_lora_scaffold)
    # ------------------------------------------------------------------

    def _run_legacy(self) -> None:
        """Run the legacy pipeline (naive_sequential / c_lora_scaffold)."""
        tasks = self.config.tasks
        num_tasks = len(tasks)

        logger.info("=" * 60)
        logger.info("CONTINUAL DREAMBOOTH-CLORA PIPELINE (LEGACY)")
        logger.info("Method: %s", self.config.experiment.method)
        logger.info("Tasks: %d", num_tasks)
        logger.info("Output: %s", self.base_output)
        if self.config.experiment.method == "c_lora_scaffold":
            logger.info("C-LoRA importance: %s", self.config.c_lora.importance_method)
        logger.info("=" * 60)

        self.validate_all_tasks()
        self.trainer.load_models()

        prev_checkpoint = None

        for t, task in enumerate(tasks):
            stage_start = time.time()
            logger.info("")
            logger.info("=" * 60)
            logger.info("TASK %d/%d: %s", t + 1, num_tasks, task.name)
            logger.info("=" * 60)

            self.method.pre_task_setup(t, self.trainer, self.config)
            self.trainer.inject_lora(prev_checkpoint=prev_checkpoint)

            # Fisher hooks for scaffold
            if (self.config.experiment.method == "c_lora_scaffold"
                    and self.config.c_lora.importance_method == "fisher_diag"
                    and t > 0):
                from methods.c_lora_scaffold import CLoRAScaffold
                if isinstance(self.method, CLoRAScaffold):
                    self.method.setup_fisher_hooks(self.trainer)

            extra_loss_fn = self.method.get_extra_loss_fn()

            task_output_dir = str(self.checkpoints_dir / f"task_{t:02d}_{task.name}")
            prev_checkpoint = self.trainer.train(
                task=task,
                output_dir=task_output_dir,
                extra_loss_fn=extra_loss_fn,
            )

            self.method.post_task_cleanup(t, self.trainer)

            logger.info("Evaluating tasks 0..%d after training task %d", t, t)
            pipe = self.trainer.build_inference_pipeline(lora_dir=prev_checkpoint)
            self._evaluate_stage(t, pipe)

            del pipe
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            save_json(self.scores, str(self.base_output / "scores_intermediate.json"))

            stage_time = time.time() - stage_start
            logger.info("Task %d complete in %.1f seconds", t, stage_time)

        self._generate_report(num_tasks)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the full continual learning pipeline."""
        if self.config.experiment.method == "faithful_c_lora":
            self._run_faithful()
        else:
            self._run_legacy()

    def run_eval_only(self, checkpoint_dir: str) -> None:
        """Run evaluation only using existing checkpoints.

        For faithful_c_lora: restores the full continual state (tokenizer,
        all adapters, token embeddings) and runs PRIMARY composed evaluation.
        For legacy: loads a single LoRA checkpoint.

        Args:
            checkpoint_dir: For faithful_c_lora: base experiment output directory.
                           For legacy: path to LoRA checkpoint directory.
        """
        tasks = self.config.tasks
        num_tasks = len(tasks)

        logger.info("=" * 60)
        logger.info("EVAL-ONLY MODE")
        logger.info("Method: %s", self.config.experiment.method)
        logger.info("Checkpoint: %s", checkpoint_dir)
        logger.info("Tasks: %d", num_tasks)
        logger.info("=" * 60)

        self.validate_all_tasks()
        self.trainer.load_models()

        if self.config.experiment.method == "faithful_c_lora":
            # Faithful restore: reconstruct the entire continual state
            self._run_eval_only_faithful(checkpoint_dir)
        else:
            # Legacy: single LoRA checkpoint
            pipe = self.trainer.build_inference_pipeline(lora_dir=checkpoint_dir)

            final_stage = num_tasks - 1
            self._evaluate_stage(final_stage, pipe)

            del pipe
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._generate_report(num_tasks)

    def _run_eval_only_faithful(self, experiment_dir: str) -> None:
        """Eval-only mode for faithful_c_lora.

        Restores the full shared continual model from saved state, then
        runs PRIMARY evaluation (composed model, all adapters active).
        Optionally also runs SECONDARY diagnostic and multi-concept eval.

        Args:
            experiment_dir: Base experiment output directory (contains
                           faithful_state/, task_registry.json, checkpoints/).
        """
        c_lora_cfg = self.config.c_lora
        tasks = self.config.tasks

        # Load task registry to know how many tasks were completed
        registry_path = Path(experiment_dir) / "task_registry.json"
        if registry_path.exists():
            registry = load_task_registry(experiment_dir)
            num_completed = registry.get("num_tasks_completed", 0)
            logger.info(
                "Loaded task registry: %d tasks completed", num_completed,
            )
        else:
            num_completed = len(tasks)
            logger.warning(
                "No task registry found, assuming all %d tasks completed",
                num_completed,
            )

        # Restore faithful state
        self.trainer.restore_faithful_state(experiment_dir)

        # PRIMARY EVAL: composed model with all adapters active
        final_stage = num_completed - 1
        logger.info(
            "PRIMARY EVAL (composed model): evaluating %d tasks at stage %d",
            num_completed, final_stage,
        )

        pipe = self.trainer.build_inference_pipeline(
            adapter_names=self.trainer.get_all_adapter_names(),
        )
        self._evaluate_stage(final_stage, pipe)

        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # SECONDARY: diagnostic eval (optional)
        if c_lora_cfg.run_diagnostic_eval:
            self._run_diagnostic_eval(final_stage)

        # Multi-concept probe (optional)
        if c_lora_cfg.run_multi_concept_probe and final_stage >= 1:
            self._run_multi_concept_probe(final_stage)

        self._generate_report(num_completed)

    def resume_faithful(self) -> None:
        """Resume faithful_c_lora training from checkpoint.

        Restores state from the last completed task and continues
        with the next uncompleted task.
        """
        if self.config.experiment.method != "faithful_c_lora":
            raise ValueError("resume_faithful only works with faithful_c_lora method")

        tasks = self.config.tasks
        c_lora_cfg = self.config.c_lora

        # Load registry to determine resume point
        registry = load_task_registry(str(self.base_output))
        num_completed = registry.get("num_tasks_completed", 0)
        self.task_registry = registry

        logger.info("=" * 60)
        logger.info("RESUME MODE: faithful_c_lora")
        logger.info("Completed tasks: %d / %d", num_completed, len(tasks))
        logger.info("=" * 60)

        if num_completed >= len(tasks):
            logger.info("All tasks already completed, nothing to resume")
            self._generate_report(len(tasks))
            return

        self.validate_all_tasks()
        self.trainer.load_models()

        # Restore state from checkpoint
        self.trainer.restore_faithful_state(str(self.base_output))

        # Rebuild occupancy masks from saved adapters (for regularization)
        from methods.faithful_c_lora import FaithfulCLoRA
        if isinstance(self.method, FaithfulCLoRA):
            for t in range(num_completed):
                adapter_name = f"task_{t}"
                factors = self.trainer.get_adapter_lora_factors(adapter_name)
                self.method._past_factors.append(factors)
                # Update occupancy masks
                for module_key, (a_mat, b_mat) in factors.items():
                    delta = torch.matmul(b_mat, a_mat)
                    abs_delta = delta.abs()
                    if module_key in self.method._occupancy:
                        self.method._occupancy[module_key] += abs_delta
                    else:
                        self.method._occupancy[module_key] = abs_delta
                # Normalize
                for module_key in self.method._occupancy:
                    occ = self.method._occupancy[module_key]
                    max_val = occ.max()
                    if max_val > 0:
                        self.method._occupancy[module_key] = occ / max_val

            logger.info(
                "Rebuilt occupancy from %d completed adapters (%d modules)",
                num_completed, len(self.method._occupancy),
            )

        # Load previous scores
        scores_path = self.base_output / "scores_intermediate.json"
        if scores_path.exists():
            from utils.io import load_json
            self.scores = {int(k): v for k, v in load_json(str(scores_path)).items()}

        # Setup prior preservation
        prior_dataset = None
        if c_lora_cfg.prior_preservation:
            class_images_dir = str(self.base_output / "class_prior")
            if Path(class_images_dir).is_dir():
                prior_dataset = PriorPreservationDataset(
                    class_images_dir=class_images_dir,
                    class_prompt=c_lora_cfg.class_prompt,
                    tokenizer=self.trainer.tokenizer,
                    size=512,
                )

        # Continue from next uncompleted task
        for t in range(num_completed, len(tasks)):
            task = tasks[t]
            stage_start = time.time()
            logger.info("")
            logger.info("=" * 60)
            logger.info("RESUME TASK %d/%d: %s", t + 1, len(tasks), task.name)
            logger.info("=" * 60)

            self.method.pre_task_setup(t, self.trainer, self.config)
            self.method.setup_adapters(t, self.trainer, self.config)

            prompt = build_task_prompt(task, c_lora_cfg.prompt_mode)
            extra_loss_fn = self.method.get_extra_loss_fn()

            token_id = self.trainer._task_token_ids.get(f"<{task.trigger_token}>")
            task_output_dir = str(
                self.checkpoints_dir / f"task_{t:02d}_{task.name}"
            )

            self.trainer.train(
                task=task,
                output_dir=task_output_dir,
                extra_loss_fn=extra_loss_fn,
                prompt_override=prompt,
                prior_dataset=prior_dataset,
                prior_loss_weight=c_lora_cfg.prior_loss_weight,
                token_ids_to_save=[token_id] if token_id else None,
            )

            self.method.post_task_cleanup(t, self.trainer)
            self.trainer.clear_token_embedding_hooks()

            # Save metadata
            task_info = {
                "task_name": task.name,
                "task_idx": t,
                "task_order": t,
                "adapter_name": f"task_{t}",
                "token": f"<{task.trigger_token}>",
                "token_id": token_id,
                "prompt_mode": c_lora_cfg.prompt_mode,
                "instance_prompt": prompt,
                "class_prompt": c_lora_cfg.class_prompt,
                "regularizer_type": c_lora_cfg.regularizer_type,
                "regularization_weight": c_lora_cfg.regularization_weight,
                "lora_rank": self.config.model.lora_rank,
                "lora_alpha": self.config.model.lora_alpha,
                "lora_target_modules": self.config.model.lora_target_modules,
                "max_train_steps": self.config.training.max_train_steps,
                "seed": self.config.experiment.seed,
                "checkpoint_dir": task_output_dir,
                "token_embedding_path": str(Path(task_output_dir) / "token_embeddings.pt"),
                "adapter_weights_path": str(Path(task_output_dir) / "lora_weights"),
            }
            save_task_info(task_output_dir, task_info)
            self.task_registry["tasks"].append(task_info)
            self.task_registry["num_tasks_completed"] = t + 1
            save_task_registry(str(self.base_output), self.task_registry)
            self.trainer.save_faithful_state(str(self.base_output))

            # Evaluate
            pipe = self.trainer.build_inference_pipeline(
                adapter_names=self.trainer.get_all_adapter_names(),
            )
            self._evaluate_stage(t, pipe)
            del pipe
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            save_json(self.scores, str(self.base_output / "scores_intermediate.json"))
            stage_time = time.time() - stage_start
            logger.info("Task %d complete in %.1f seconds", t, stage_time)

        self._generate_report(len(tasks))

    def _generate_report(self, num_tasks: int) -> None:
        """Build and save the final report from accumulated scores."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("GENERATING FINAL REPORT")
        logger.info("=" * 60)

        task_names = [task.name for task in self.config.tasks]
        score_matrix = build_score_matrix(self.scores, num_tasks)
        metrics = compute_forgetting_metrics(score_matrix)

        save_full_report(
            score_matrix=score_matrix,
            metrics=metrics,
            task_names=task_names,
            output_dir=str(self.base_output),
            experiment_name=self.config.experiment.name,
            confusion_data=self.confusion_data if self.confusion_data else None,
            per_prompt_data=self.per_prompt_data if self.per_prompt_data else None,
        )

        logger.info("Pipeline complete! Results saved to %s", self.base_output)
