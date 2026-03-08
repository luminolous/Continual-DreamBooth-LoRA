"""
Abstract base class for continual learning methods.

Each method provides hooks that the orchestrator calls around the training loop
for each task. This allows different continual-learning strategies to modify
training behaviour without changing the core training code.

Lifecycle per task:
    1. pre_task_setup(task_idx, trainer, config)
    2. setup_adapters(task_idx, trainer, config)   [optional, default no-op]
    3. get_extra_loss_fn() -> Optional[Callable]
    4. <training runs>
    5. post_task_cleanup(task_idx, trainer)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import torch


class BaseMethod(ABC):
    """Interface for continual learning methods.

    Subclasses implement hooks called by the orchestrator at specific
    points in the per-task training lifecycle.
    """

    @abstractmethod
    def pre_task_setup(
        self,
        task_idx: int,
        trainer: Any,
        config: Any,
    ) -> None:
        """Called before training on each task.

        Use this to snapshot weights, build regularization references, etc.

        Args:
            task_idx: Zero-based index of the current task.
            trainer: The DreamBoothLoRATrainer instance.
            config: The full PipelineConfig.
        """
        ...

    def setup_adapters(
        self,
        task_idx: int,
        trainer: Any,
        config: Any,
    ) -> None:
        """Called after pre_task_setup, before training begins.

        Use this to create/configure task-specific adapters.
        Default implementation is a no-op (backward compatible).

        Args:
            task_idx: Zero-based index of the current task.
            trainer: The DreamBoothLoRATrainer instance.
            config: The full PipelineConfig.
        """
        pass  # No-op default for naive_sequential and c_lora_scaffold

    @abstractmethod
    def get_extra_loss_fn(self) -> Optional[Callable[[torch.nn.Module], torch.Tensor]]:
        """Return an optional extra loss function for regularization.

        The returned callable receives the UNet and should return a scalar
        loss tensor that gets added to the DreamBooth loss.

        Returns:
            A callable(unet) -> loss_tensor, or None for no extra loss.
        """
        ...

    @abstractmethod
    def post_task_cleanup(
        self,
        task_idx: int,
        trainer: Any,
    ) -> None:
        """Called after training on each task.

        Use this to archive adapter factors, update importance, free memory.

        Args:
            task_idx: Zero-based index of the completed task.
            trainer: The DreamBoothLoRATrainer instance.
        """
        ...
