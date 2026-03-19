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

from config.schema import TaskConfig

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

def build_eval_prompt(task: TaskConfig, prompt_mode: Optional[str] = None) -> str:
    return task.eval_prompt or task.instance_prompt or f"a portrait of <{task.trigger_token}>"

# ---------------------------------------------------------------------------
# Instance dataset
# ---------------------------------------------------------------------------

import json
import random
import re
from typing import Dict

def _candidate_caption_paths(image_path: Path, exts: List[str]) -> List[Path]:
    paths = []
    for ext in exts:
        ext = ext if ext.startswith(".") else f".{ext}"
        paths.append(image_path.with_suffix(ext))                  # image.png -> image.txt
        paths.append(image_path.parent / f"{image_path.name}{ext}")  # image.png.txt
    return paths

def _load_metadata_map(data_dir: Path) -> Dict[str, str]:
    candidates = [
        data_dir / "metadata.jsonl",
        data_dir / "captions.jsonl",
        data_dir / "tags.jsonl",
    ]
    mapping: Dict[str, str] = {}
    for path in candidates:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                fname = (
                    obj.get("file_name")
                    or obj.get("filename")
                    or obj.get("image")
                    or obj.get("path")
                )
                text = (
                    obj.get("text")
                    or obj.get("caption")
                    or obj.get("tags")
                    or obj.get("prompt")
                )
                if fname and text:
                    mapping[Path(fname).name] = str(text)
        break
    return mapping

def _split_tags(raw: str) -> List[str]:
    parts = re.split(r"[,\\n|]", raw)
    return [p.strip() for p in parts if p.strip()]

def _sanitize_tags(tags: List[str], task: TaskConfig, cfg) -> List[str]:
    out = []
    trigger_plain = task.trigger_token.lower().strip()
    task_name = task.name.lower().strip()

    banned = {f"<{trigger_plain}>", trigger_plain}
    if cfg.strip_task_name_from_tags and task_name:
        banned.add(task_name)

    for tag in tags:
        t = tag.strip()
        t_cmp = t.lower() if cfg.lowercase_tags else t
        if t_cmp in banned:
            continue
        if cfg.lowercase_tags:
            t = t_cmp
        out.append(t)

    # deduplicate while preserving order
    dedup = []
    seen = set()
    for t in out:
        if t not in seen:
            dedup.append(t)
            seen.add(t)

    if cfg.shuffle_tags:
        random.shuffle(dedup)

    if cfg.max_caption_tags is not None:
        dedup = dedup[: cfg.max_caption_tags]

    return dedup

def build_image_caption_prompt(task: TaskConfig, raw_caption: str, cfg) -> str:
    prefix = cfg.caption_prefix_template.format(
        trigger_token=task.trigger_token,
        name=task.name,
    ).strip()

    suffix = cfg.caption_suffix_template.format(
        trigger_token=task.trigger_token,
        name=task.name,
    ).strip()

    tags = _sanitize_tags(_split_tags(raw_caption), task, cfg)
    body = ", ".join(tags)

    pieces = []
    if prefix:
        pieces.append(prefix)
    if body:
        pieces.append(body)
    if suffix:
        pieces.append(suffix)

    return ", ".join([p for p in pieces if p]).strip(", ")

class TaggedConceptDataset(Dataset):
    def __init__(
        self,
        task: TaskConfig,
        tokenizer,
        c_lora_config,
        size: int = 512,
        center_crop: bool = True,
        repeats: int = 1,
    ):
        self.task = task
        self.tokenizer = tokenizer
        self.size = size
        self.c_lora_config = c_lora_config
        self.data_dir = Path(task.data_dir)

        self.image_paths = find_images(self.data_dir)
        if not self.image_paths:
            raise ValueError(f"No images found in {self.data_dir}")

        self.image_paths = self.image_paths * max(1, repeats)
        self.metadata_map = _load_metadata_map(self.data_dir)

        transform_list = [transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)]
        if center_crop:
            transform_list.append(transforms.CenterCrop(size))
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.transform = transforms.Compose(transform_list)

        logger.info(
            "TaggedConceptDataset: %d images (x%d repeats) from %s",
            len(find_images(self.data_dir)),
            repeats,
            self.data_dir,
        )

    def _read_caption(self, image_path: Path) -> str:
        for cand in _candidate_caption_paths(
            image_path,
            self.c_lora_config.caption_extensions,
        ):
            if cand.exists():
                return cand.read_text(encoding="utf-8").strip()

        meta_caption = self.metadata_map.get(image_path.name, "").strip()
        if meta_caption:
            return meta_caption

        # fallback: token only
        return ""

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict:
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.transform(image)

        raw_caption = self._read_caption(image_path)
        prompt = build_image_caption_prompt(
            self.task,
            raw_caption=raw_caption,
            cfg=self.c_lora_config,
        )

        input_ids = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "prompt": prompt,
            "image_path": str(image_path),
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


def collate_examples(batch: List[dict]) -> dict:
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    input_ids = torch.stack([b["input_ids"] for b in batch])
    output = {"pixel_values": pixel_values, "input_ids": input_ids}
    if "prompt" in batch[0]:
        output["prompts"] = [b["prompt"] for b in batch]
    if "image_path" in batch[0]:
        output["image_paths"] = [b["image_path"] for b in batch]
    return output
