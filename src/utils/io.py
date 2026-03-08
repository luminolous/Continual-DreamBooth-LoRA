"""I/O helpers for checkpoints, LoRA weights, token embeddings, task registry,
and directory management."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it doesn't exist. Returns the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# LoRA weight save/load
# ---------------------------------------------------------------------------

def save_lora_weights(
    unet: torch.nn.Module,
    output_dir: str | Path,
    filename: str = "lora_weights.safetensors",
) -> Path:
    """Save LoRA adapter weights from a PEFT-wrapped UNet.

    Args:
        unet: The PEFT-wrapped UNet model.
        output_dir: Directory to save weights to.
        filename: Name of the weights file.

    Returns:
        Path to the saved weights directory.
    """
    from peft import PeftModel

    out_path = ensure_dir(output_dir)

    if isinstance(unet, PeftModel):
        unet.save_pretrained(str(out_path))
        logger.info("Saved LoRA weights (PEFT) to %s", out_path)
    else:
        # Fallback: save only lora parameters by name
        lora_state = {
            k: v.cpu().clone()
            for k, v in unet.state_dict().items()
            if "lora" in k.lower()
        }
        save_path = out_path / filename
        torch.save(lora_state, str(save_path))
        logger.info("Saved LoRA state dict (%d params) to %s", len(lora_state), save_path)

    return out_path


def load_lora_weights_into_pipeline(
    pipeline,
    lora_dir: str | Path,
    adapter_name: str = "default",
) -> None:
    """Load LoRA weights into a Diffusers pipeline.

    Uses pipeline.load_lora_weights which supports both safetensors and
    PEFT adapter directories.

    Args:
        pipeline: A Diffusers StableDiffusionPipeline instance.
        lora_dir: Directory containing the saved LoRA weights.
        adapter_name: Name for the LoRA adapter.
    """
    lora_path = Path(lora_dir)
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA weights directory not found: {lora_path}")

    pipeline.load_lora_weights(str(lora_path), adapter_name=adapter_name)
    logger.info("Loaded LoRA weights from %s (adapter=%s)", lora_path, adapter_name)


# ---------------------------------------------------------------------------
# Token embedding save/load
# ---------------------------------------------------------------------------

def save_token_embeddings(
    text_encoder: torch.nn.Module,
    token_ids: List[int],
    path: str | Path,
) -> None:
    """Save learned token embeddings to disk.

    Saves a dict mapping token_id -> embedding_vector for the specified tokens.

    Args:
        text_encoder: The text encoder model (CLIPTextModel).
        token_ids: List of token IDs whose embeddings to save.
        path: File path to save to (.pt file).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    embeddings = text_encoder.get_input_embeddings()
    token_data = {}
    for tid in token_ids:
        token_data[tid] = embeddings.weight[tid].detach().cpu().clone()

    torch.save(token_data, str(path))
    logger.info("Saved %d token embeddings to %s", len(token_data), path)


def load_token_embeddings(
    text_encoder: torch.nn.Module,
    tokenizer,
    path: str | Path,
) -> None:
    """Load saved token embeddings into a text encoder.

    The tokenizer must already have the tokens added (via add_tokens)
    and the text encoder must have been resized accordingly.

    Args:
        text_encoder: The text encoder model.
        tokenizer: The tokenizer (must already contain the tokens).
        path: File path to load from (.pt file).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Token embeddings file not found: {path}")

    token_data = torch.load(str(path), map_location="cpu", weights_only=True)
    embeddings = text_encoder.get_input_embeddings()

    for tid, emb_vector in token_data.items():
        tid = int(tid)
        if tid < embeddings.weight.shape[0]:
            embeddings.weight.data[tid] = emb_vector.to(embeddings.weight.device)
        else:
            logger.warning(
                "Token ID %d out of range (embedding size %d), skipping",
                tid, embeddings.weight.shape[0],
            )

    logger.info("Loaded %d token embeddings from %s", len(token_data), path)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

def save_task_registry(
    output_dir: str | Path,
    registry: Dict[str, Any],
) -> None:
    """Save the global task registry JSON.

    The registry maps experiment metadata and the list of completed tasks
    with their adapter names, tokens, prompt modes, and checkpoint paths.

    Args:
        output_dir: Base experiment output directory.
        registry: The registry dict to save.
    """
    path = Path(output_dir) / "task_registry.json"
    save_json(registry, path)
    logger.info("Saved task registry (%d tasks) to %s",
                len(registry.get("tasks", [])), path)


def load_task_registry(output_dir: str | Path) -> Dict[str, Any]:
    """Load the global task registry JSON.

    Args:
        output_dir: Base experiment output directory.

    Returns:
        The registry dict.
    """
    path = Path(output_dir) / "task_registry.json"
    return load_json(path)


def save_task_info(
    task_dir: str | Path,
    task_info: Dict[str, Any],
) -> None:
    """Save per-task metadata JSON.

    Args:
        task_dir: The task's checkpoint directory.
        task_info: Dict with task_name, adapter_name, token, etc.
    """
    path = Path(task_dir) / "task_info.json"
    save_json(task_info, path)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def save_json(data: Any, path: str | Path) -> None:
    """Save data as formatted JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str | Path) -> Dict:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
