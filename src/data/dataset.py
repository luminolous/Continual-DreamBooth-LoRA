"""
Dataset classes and prompt building for Continual DreamBooth-CLoRA.

Handles:
- Directory validation (existence, minimum image count, readability)
- Image loading with SD1.5-compatible transforms
- Pairing pixel values with tokenized prompts
- Task prompt construction for dreambooth vs clora prompt modes
- Prior preservation class images dataset
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.config.schema import TaskConfig

logger = logging.getLogger(__name__)

# Extensions recognized as images
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def find_images(directory: str | Path) -> List[Path]:
    """Find all image files in a directory (non-recursive).

    Args:
        directory: Path to search for images.

    Returns:
        Sorted list of image file paths.
    """
    d = Path(directory)
    if not d.is_dir():
        return []
    return sorted(
        p for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_task_data(task: TaskConfig, min_images: int = 1) -> None:
    """Validate that a task's dataset directories are well-formed.

    Args:
        task: Task configuration to validate.
        min_images: Minimum required training images.

    Raises:
        FileNotFoundError: If data_dir or ref_dir doesn't exist.
        ValueError: If insufficient images found.
    """
    data_path = Path(task.data_dir)
    ref_path = Path(task.ref_dir)

    if not data_path.is_dir():
        raise FileNotFoundError(
            f"Task '{task.name}': training data directory not found: {data_path}"
        )
    if not ref_path.is_dir():
        raise FileNotFoundError(
            f"Task '{task.name}': reference directory not found: {ref_path}"
        )

    train_images = find_images(data_path)
    if len(train_images) < min_images:
        raise ValueError(
            f"Task '{task.name}': found {len(train_images)} training images "
            f"in {data_path}, need at least {min_images}"
        )

    ref_images = find_images(ref_path)
    if len(ref_images) < 1:
        raise ValueError(
            f"Task '{task.name}': no reference images found in {ref_path}"
        )

    logger.info(
        "Task '%s': %d training images, %d reference images",
        task.name, len(train_images), len(ref_images),
    )


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_task_prompt(task: TaskConfig, prompt_mode: str) -> str:
    """Build the training/eval prompt for a task based on prompt mode.

    Args:
        task: Task configuration with trigger_token and instance_prompt.
        prompt_mode: "clora" or "dreambooth".

    Returns:
        The constructed prompt string.

    In clora mode: "a photo of <trigger_token>" — no explicit class word.
    In dreambooth mode: uses the existing instance_prompt as-is.
    """
    if prompt_mode == "clora":
        # C-LoRA-style: personalized token only, no object/class word
        return f"a photo of <{task.trigger_token}>"
    elif prompt_mode == "dreambooth":
        # DreamBooth-style: uses the full instance_prompt from config
        return task.instance_prompt
    else:
        raise ValueError(f"Unknown prompt_mode '{prompt_mode}', expected 'clora' or 'dreambooth'")


# ---------------------------------------------------------------------------
# Instance dataset
# ---------------------------------------------------------------------------

class DreamBoothDataset(Dataset):
    """Dataset for DreamBooth-LoRA training on a single character.

    Loads images from a directory and pairs them with tokenized instance prompts.
    Images are resized and normalized for Stable Diffusion's VAE.
    """

    def __init__(
        self,
        data_dir: str,
        instance_prompt: str,
        tokenizer,
        size: int = 512,
        center_crop: bool = True,
        repeats: int = 1,
    ):
        """
        Args:
            data_dir: Directory containing training images.
            instance_prompt: The instance prompt string.
            tokenizer: HuggingFace tokenizer for the text encoder.
            size: Target image size (square).
            center_crop: Whether to center-crop images.
            repeats: Number of times to repeat the dataset (for short datasets).
        """
        self.data_dir = Path(data_dir)
        self.instance_prompt = instance_prompt
        self.tokenizer = tokenizer
        self.size = size

        self.image_paths = find_images(self.data_dir)
        if not self.image_paths:
            raise ValueError(f"No images found in {self.data_dir}")

        # Repeat short datasets to fill an epoch
        self.image_paths = self.image_paths * max(1, repeats)

        # Build transform pipeline
        transform_list = [transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)]
        if center_crop:
            transform_list.append(transforms.CenterCrop(size))
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # Scale to [-1, 1]
        ])
        self.transform = transforms.Compose(transform_list)

        # Pre-tokenize the instance prompt
        self.input_ids = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        logger.info(
            "DreamBoothDataset: %d images (×%d repeats) from %s, prompt='%s'",
            len(find_images(self.data_dir)), repeats, self.data_dir, instance_prompt,
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict:
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.transform(image)

        return {
            "pixel_values": pixel_values,
            "input_ids": self.input_ids.clone(),
        }


# ---------------------------------------------------------------------------
# Prior preservation dataset
# ---------------------------------------------------------------------------

class PriorPreservationDataset(Dataset):
    """Dataset of class-prior images for prior preservation regularization.

    Loads pregenerated class images (from the base model) and pairs them
    with tokenized class prompts. Used alongside DreamBoothDataset during
    training to prevent language drift.
    """

    def __init__(
        self,
        class_images_dir: str,
        class_prompt: str,
        tokenizer,
        size: int = 512,
    ):
        """
        Args:
            class_images_dir: Directory containing pregenerated class images.
            class_prompt: The class prompt (e.g., "a photo of anime character").
            tokenizer: HuggingFace tokenizer.
            size: Target image size (square).
        """
        self.class_images_dir = Path(class_images_dir)
        self.class_prompt = class_prompt

        self.image_paths = find_images(self.class_images_dir)
        if not self.image_paths:
            raise ValueError(
                f"No class prior images found in {self.class_images_dir}. "
                "Generate them first using the base model."
            )

        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.input_ids = tokenizer(
            class_prompt,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        logger.info(
            "PriorPreservationDataset: %d class images from %s, prompt='%s'",
            len(self.image_paths), self.class_images_dir, class_prompt,
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict:
        image_path = self.image_paths[index % len(self.image_paths)]
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.transform(image)

        return {
            "pixel_values": pixel_values,
            "input_ids": self.input_ids.clone(),
        }


def collate_dreambooth(batch: List[dict]) -> dict:
    """Custom collate function for DreamBooth batches."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    input_ids = torch.stack([b["input_ids"] for b in batch])
    return {"pixel_values": pixel_values, "input_ids": input_ids}
