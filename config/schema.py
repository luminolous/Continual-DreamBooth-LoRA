"""
Configuration schema for Continual DreamBooth-CLoRA experiments.

Uses Python dataclasses + PyYAML for simplicity.
All paths in the config are resolved relative to the config file location.

Supports three methods:
  - naive_sequential: no continual regularization (forgetting baseline)
  - c_lora_scaffold: legacy L2/importance scaffold (backward compat)
  - faithful_c_lora: shared continual model with occupancy regularization
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


# ---------------------------------------------------------------------------
# Sub-config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    name: str = "experiment"
    seed: int = 42
    output_dir: str = "./outputs"
    method: str = "naive_sequential"  # "naive_sequential" | "c_lora_scaffold" | "faithful_c_lora"


@dataclass
class ModelConfig:
    pretrained_model_name: str = "runwayml/stable-diffusion-v1-5"
    lora_rank: int = 4
    lora_alpha: int = 4
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["to_k", "to_v"]
    )
    use_xformers: bool = False
    enable_xformers_memory_efficient_attention: bool = False
    mixed_precision: str = "fp16"


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    max_train_steps: int = 400
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    use_8bit_adam: bool = False
    pre_compute_text_embeddings: bool = False
    prior_preservation: bool = False  # legacy field for naive/scaffold
    seed: int = 42


@dataclass
class TaskConfig:
    name: str = ""
    trigger_token: str = ""
    instance_prompt: str = ""
    eval_prompt: str = ""
    class_prompt: str = "a photo of anime character"
    data_dir: str = ""
    ref_dir: str = ""


@dataclass
class EvaluationConfig:
    num_images_per_prompt: int = 4
    guidance_scale: float = 7.5
    num_inference_steps: int = 30
    image_size: int = 512
    compute_confusion_gap: bool = False
    per_prompt_breakdown: bool = False
    prompts_per_character: List[str] = field(
        default_factory=lambda: [
            "{instance_prompt}, portrait, white background",
            "{instance_prompt}, upper body, smiling",
            "{instance_prompt}, full body, standing",
            "{instance_prompt}, close-up face, detailed",
            "{instance_prompt}, looking to the side, soft lighting",
            "{instance_prompt}, chibi style, cute expression",
        ]
    )


@dataclass
class CLoRAConfig:
    # --- Token ---
    token_init: str = "random"
    train_token_embeddings: bool = True

    # --- Prompt / caption ---
    prompt_mode: str = "clora"  # keep for compatibility/logging
    use_image_captions: bool = True
    caption_extensions: List[str] = field(
        default_factory=lambda: [".txt", ".caption", ".tags"]
    )
    caption_prefix_template: str = "a portrait of <{trigger_token}>"
    caption_suffix_template: str = ""
    strip_trigger_from_tags: bool = True
    strip_task_name_from_tags: bool = True
    lowercase_tags: bool = True
    shuffle_tags: bool = False
    max_caption_tags: Optional[int] = 48

    # --- Adapter ---
    adapter_strategy: str = "per_task"

    # --- Regularization ---
    regularizer_type: str = "occupancy"
    regularization_weight: float = 0.1

    # --- Prior preservation ---
    prior_preservation: bool = False   # set False if you want pure PEFT, no DreamBooth-like prior
    prior_loss_weight: float = 1.0
    num_class_images: int = 200
    class_prompt: str = "a photo of anime character"

    # --- Inference ---
    inference_adapter_mode: str = "merged_online"

    # --- Diagnostic ---
    run_diagnostic_eval: bool = False
    run_multi_concept_probe: bool = True

    # --- Legacy scaffold ---
    importance_method: str = "magnitude"
    importance_decay: float = 0.9
    regularize_layers: List[str] = field(default_factory=lambda: ["to_k", "to_v"])


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Root configuration for a Continual DreamBooth-CLoRA experiment."""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tasks: List[TaskConfig] = field(default_factory=list)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    c_lora: CLoRAConfig = field(default_factory=CLoRAConfig)

    # Internal: absolute path to the config file's parent (set after loading)
    _config_dir: str = ""


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def _dict_to_dataclass(cls, data: dict):
    """Recursively convert a dict to a dataclass, ignoring unknown keys."""
    if data is None:
        return cls()
    field_names = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered)


def load_config(path: str) -> PipelineConfig:
    """Load and validate a YAML configuration file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A fully populated PipelineConfig instance.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If required fields are missing or invalid.
    """
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Config file is empty: {config_path}")

    config_dir = str(config_path.parent)

    # Parse each section
    experiment = _dict_to_dataclass(ExperimentConfig, raw.get("experiment", {}))
    model = _dict_to_dataclass(ModelConfig, raw.get("model", {}))
    training = _dict_to_dataclass(TrainingConfig, raw.get("training", {}))
    evaluation = _dict_to_dataclass(EvaluationConfig, raw.get("evaluation", {}))
    c_lora = _dict_to_dataclass(CLoRAConfig, raw.get("c_lora", {}))

    # Parse tasks list
    tasks_raw = raw.get("tasks", [])
    if not tasks_raw:
        raise ValueError("Config must define at least one task under 'tasks:'")

    tasks = [_dict_to_dataclass(TaskConfig, t) for t in tasks_raw]

    # Validate tasks
    for i, task in enumerate(tasks):
        if not task.name:
            raise ValueError(f"Task {i} is missing 'name'")
        if not task.trigger_token:
            raise ValueError(f"Task '{task.name}' is missing 'trigger_token'")
        if not task.eval_prompt and not task.instance_prompt:
            task.eval_prompt = f"a portrait of <{task.trigger_token}>"
        if not task.data_dir:
            raise ValueError(f"Task '{task.name}' is missing 'data_dir'")
        if not task.ref_dir:
            raise ValueError(f"Task '{task.name}' is missing 'ref_dir'")

    # Resolve relative paths against config file directory
    output_dir = experiment.output_dir
    if not os.path.isabs(output_dir):
        output_dir = str(Path(config_dir) / output_dir)
    experiment.output_dir = output_dir

    for task in tasks:
        if not os.path.isabs(task.data_dir):
            task.data_dir = str(Path(config_dir) / task.data_dir)
        if not os.path.isabs(task.ref_dir):
            task.ref_dir = str(Path(config_dir) / task.ref_dir)

    # Validate method name
    valid_methods = {"naive_sequential", "c_lora_scaffold", "faithful_c_lora"}
    if experiment.method not in valid_methods:
        raise ValueError(
            f"Unknown method '{experiment.method}'. "
            f"Valid methods: {sorted(valid_methods)}"
        )

    # Method-specific validation
    if experiment.method == "faithful_c_lora":
        model.lora_target_modules = ["attn2.to_k", "attn2.to_v"]

    config = PipelineConfig(
        experiment=experiment,
        model=model,
        training=training,
        tasks=tasks,
        evaluation=evaluation,
        c_lora=c_lora,
        _config_dir=config_dir,
    )

    return config
