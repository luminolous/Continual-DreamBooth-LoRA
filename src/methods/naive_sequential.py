"""
Naive sequential baseline: no continual-learning regularization.

Simply continues DreamBooth-LoRA training from one task to the next,
loading the previous checkpoint and fine-tuning on the new character data.
This is the expected-to-forget baseline for comparison.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import torch

from src.methods.base import BaseMethod

logger = logging.getLogger(__name__)


class NaiveSequential(BaseMethod):
    """Naive sequential fine-tuning — no regularization at all.

    This serves as the catastrophic-forgetting baseline:
    - Pre-task: no-op
    - Extra loss: None
    - Post-task: no-op
    """

    def pre_task_setup(
        self,
        task_idx: int,
        trainer: Any,
        config: Any,
    ) -> None:
        logger.info(
            "[NaiveSequential] Task %d: no continual-learning setup needed", task_idx
        )

    def get_extra_loss_fn(self) -> Optional[Callable[[torch.nn.Module], torch.Tensor]]:
        return None

    def post_task_cleanup(
        self,
        task_idx: int,
        trainer: Any,
    ) -> None:
        logger.info(
            "[NaiveSequential] Task %d: no post-task cleanup needed", task_idx
        )
