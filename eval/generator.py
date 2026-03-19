"""
Evaluation image generation from LoRA-loaded Stable Diffusion pipeline.

Generates images for each evaluation prompt template, expanding variables
like {instance_prompt} or {trigger_token} per character, and saves them
to the output directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import torch

from config.schema import EvaluationConfig, TaskConfig
from data.dataset import build_eval_prompt
from utils.io import ensure_dir

logger = logging.getLogger(__name__)


def generate_eval_images(
    pipeline,
    task: TaskConfig,
    eval_config: EvaluationConfig,
    output_dir: str,
    seed: int = 42,
) -> List[Path]:
    """Generate evaluation images for a single character task.

    Args:
        pipeline: A loaded StableDiffusionPipeline with LoRA weights.
        task: The task to evaluate (used for prompt expansion).
        eval_config: Evaluation settings (num images, guidance, etc.).
        output_dir: Directory to save generated images.
        seed: Random seed for reproducibility.

    Returns:
        List of paths to generated images.
    """
    out_path = ensure_dir(output_dir)
    generated_paths: List[Path] = []
    image_idx = 0

    pipeline.set_progress_bar_config(disable=True)

    for prompt_idx, prompt_template in enumerate(eval_config.prompts_per_character):
        base_eval_prompt = build_eval_prompt(task)
        trigger_token = f"<{task.trigger_token}>"

        prompt = prompt_template.format(
            instance_prompt=base_eval_prompt,
            trigger_token=trigger_token,
            trigger_token_plain=task.trigger_token,
            name=task.name,
        )

        logger.info(
            "Generating %d images for prompt %d/%d: '%s'",
            eval_config.num_images_per_prompt,
            prompt_idx + 1,
            len(eval_config.prompts_per_character),
            prompt,
        )

        for img_i in range(eval_config.num_images_per_prompt):
            generator = torch.Generator(device=pipeline.device).manual_seed(
                seed + image_idx
            )

            with torch.no_grad():
                result = pipeline(
                    prompt=prompt,
                    num_inference_steps=eval_config.num_inference_steps,
                    guidance_scale=eval_config.guidance_scale,
                    height=eval_config.image_size,
                    width=eval_config.image_size,
                    generator=generator,
                )

            image = result.images[0]
            filename = f"{task.name}_p{prompt_idx:02d}_i{img_i:02d}.png"
            save_path = out_path / filename
            image.save(save_path)
            generated_paths.append(save_path)
            image_idx += 1

    logger.info(
        "Generated %d evaluation images for '%s' -> %s",
        len(generated_paths), task.name, out_path,
    )
    return generated_paths