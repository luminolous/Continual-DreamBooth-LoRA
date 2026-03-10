"""
C-LoRA-inspired regularization scaffold with importance-weighted penalties.

=== IMPORTANT DISCLAIMERS ===

1. This is NOT a faithful reproduction of the C-LoRA paper
   (Smith et al., "Continual Customization of Text-to-Image Diffusion
   with C-LoRA", TMLR 2024).

2. The original C-LoRA uses:
   - Importance-weighted delta masking on LoRA weight updates
   - Randomly initialized text embeddings (no object name in prompt)
   - Targeted updates to K-V cross-attention projections only

3. This implementation uses:
   - DreamBooth-style prompts with fixed trigger tokens
   - Configurable target layers (default: to_k, to_v)
   - Importance-weighted L2 regularization (Phase 2 upgrade)

4. Prompt design differs from the paper — see README for details.

=== IMPORTANCE METHODS ===

- "none":        Uniform L2 (Phase 1 behaviour): L_reg = λ * Σ ||θ - θ_prev||²
- "magnitude":   Weight by accumulated parameter change magnitude:
                 L_reg = λ * Σ Ω_i * ||θ_i - θ_i^prev||²
                 where Ω_i tracks how much each parameter changed across tasks
- "fisher_diag": Approximate diagonal Fisher penalty:
                 L_reg = λ * Σ F_i * ||θ_i - θ_i^prev||²
                 where F_i ≈ E[(∂L/∂θ_i)²] estimated from training gradients
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

import torch

from methods.base import BaseMethod

logger = logging.getLogger(__name__)


class CLoRAScaffold(BaseMethod):
    """C-LoRA-inspired regularization for continual DreamBooth-LoRA.

    Phase 2 supports importance-weighted regularization where parameters
    that were important to previous tasks get stronger protection from
    modification during subsequent tasks.

    This is a C-LoRA-inspired scaffold — not a faithful paper reproduction.
    """

    def __init__(
        self,
        regularization_weight: float = 0.1,
        importance_method: str = "magnitude",
        importance_decay: float = 0.9,
    ):
        """
        Args:
            regularization_weight: Global regularization strength (λ).
            importance_method: How to compute per-parameter importance.
                "none" = uniform L2 (Phase 1 fallback).
                "magnitude" = accumulated parameter change magnitudes.
                "fisher_diag" = diagonal Fisher information approximation.
            importance_decay: Exponential decay for importance accumulation
                across tasks. 1.0 = no decay (equal weight to all past tasks),
                0.0 = only most recent task matters.
        """
        self.regularization_weight = regularization_weight
        self.importance_method = importance_method
        self.importance_decay = importance_decay

        self._prev_weights: Dict[str, torch.Tensor] = {}
        self._importance: Dict[str, torch.Tensor] = {}
        self._fisher_accum: Dict[str, torch.Tensor] = {}
        self._active = False

        # For fisher: store gradient accumulator during training
        self._grad_accumulator: Dict[str, torch.Tensor] = {}
        self._grad_steps: int = 0

    def pre_task_setup(
        self,
        task_idx: int,
        trainer: Any,
        config: Any,
    ) -> None:
        """Snapshot LoRA weights and compute importance from the previous task."""
        if task_idx == 0:
            self._active = False
            logger.info(
                "[CLoRAScaffold] Task 0: first task, no regularization active"
            )
            return

        # Snapshot current LoRA weights (these are from the previous task)
        self._prev_weights = trainer.snapshot_lora_weights()

        # Update importance scores based on method
        if self.importance_method == "magnitude":
            self._update_magnitude_importance(trainer)
        elif self.importance_method == "fisher_diag":
            self._finalize_fisher_importance()
        # "none" = no importance tracking needed

        self._active = True

        logger.info(
            "[CLoRAScaffold] Task %d: snapshotted %d LoRA params "
            "(λ=%.4f, method=%s, decay=%.2f)",
            task_idx,
            len(self._prev_weights),
            self.regularization_weight,
            self.importance_method,
            self.importance_decay,
        )

    def _update_magnitude_importance(self, trainer: Any) -> None:
        """Compute importance as accumulated magnitude of parameter changes.

        For each parameter, importance is proportional to how much it changed
        during the most recent task. This is accumulated across tasks with
        exponential decay.

        Intuition: parameters that changed a lot to learn previous tasks are
        important to those tasks and should be protected.
        """
        current_params = trainer.get_lora_params()

        for name, param in current_params.items():
            current_val = param.detach().cpu()

            if name in self._prev_weights:
                # Delta = how much this parameter changed during the last task
                delta = (current_val - self._prev_weights[name]).abs()
            else:
                # First task: just use the absolute magnitude
                delta = current_val.abs()

            # Normalize delta to [0, 1] range per parameter tensor
            if delta.max() > 0:
                delta = delta / delta.max()

            # Accumulate with exponential decay
            if name in self._importance:
                self._importance[name] = (
                    self.importance_decay * self._importance[name] + delta
                )
            else:
                self._importance[name] = delta

        logger.info(
            "[CLoRAScaffold] Updated magnitude importance for %d parameters",
            len(self._importance),
        )

    def setup_fisher_hooks(self, trainer: Any) -> None:
        """Install gradient accumulation hooks for Fisher diagonal estimation.

        Call this BEFORE training starts on each task (except the first).
        The hooks accumulate squared gradients throughout training.

        NOTE: This is called by the orchestrator when importance_method="fisher_diag".
        """
        self._grad_accumulator.clear()
        self._grad_steps = 0

        lora_params = trainer.get_lora_params()
        for name, param in lora_params.items():
            self._grad_accumulator[name] = torch.zeros_like(param, device="cpu")

        def _hook_factory(param_name):
            def _hook(grad):
                if grad is not None:
                    self._grad_accumulator[param_name] += (
                        grad.detach().cpu() ** 2
                    )
                    self._grad_steps += 1
            return _hook

        for name, param in lora_params.items():
            param.register_hook(_hook_factory(name))

        logger.info(
            "[CLoRAScaffold] Installed Fisher gradient hooks on %d parameters",
            len(lora_params),
        )

    def _finalize_fisher_importance(self) -> None:
        """Compute Fisher importance from accumulated squared gradients."""
        if not self._grad_accumulator or self._grad_steps == 0:
            logger.warning(
                "[CLoRAScaffold] No Fisher gradients accumulated. "
                "Falling back to uniform importance."
            )
            return

        steps_per_param = self._grad_steps / max(len(self._grad_accumulator), 1)

        for name, grad_sq_sum in self._grad_accumulator.items():
            # Average squared gradient = diagonal Fisher approximation
            fisher_diag = grad_sq_sum / max(steps_per_param, 1.0)

            # Normalize
            if fisher_diag.max() > 0:
                fisher_diag = fisher_diag / fisher_diag.max()

            # Accumulate with decay
            if name in self._importance:
                self._importance[name] = (
                    self.importance_decay * self._importance[name] + fisher_diag
                )
            else:
                self._importance[name] = fisher_diag

        logger.info(
            "[CLoRAScaffold] Finalized Fisher importance from %d accumulated "
            "gradient steps across %d parameters",
            self._grad_steps, len(self._grad_accumulator),
        )

        # Reset accumulator for next task
        self._grad_accumulator.clear()
        self._grad_steps = 0

    def get_extra_loss_fn(self) -> Optional[Callable[[torch.nn.Module], torch.Tensor]]:
        """Return importance-weighted L2 regularization loss."""
        if not self._active or not self._prev_weights:
            return None

        prev_weights = self._prev_weights
        importance = self._importance
        reg_weight = self.regularization_weight
        use_importance = self.importance_method != "none" and bool(importance)

        def _compute_reg_loss(unet: torch.nn.Module) -> torch.Tensor:
            device = next(unet.parameters()).device
            reg_loss = torch.tensor(0.0, device=device)
            matched = 0

            for name, param in unet.named_parameters():
                if not param.requires_grad:
                    continue
                if name not in prev_weights:
                    continue

                prev = prev_weights[name].to(device)
                diff_sq = (param - prev) ** 2

                if use_importance and name in importance:
                    # Importance-weighted: protect params important to past tasks
                    imp = importance[name].to(device)
                    reg_loss = reg_loss + torch.sum(imp * diff_sq)
                else:
                    # Uniform L2 fallback
                    reg_loss = reg_loss + torch.sum(diff_sq)

                matched += 1

            if matched == 0:
                logger.warning(
                    "[CLoRAScaffold] No matching parameters found for "
                    "regularization. Check parameter names."
                )
                return torch.tensor(0.0, device=device)

            return reg_weight * reg_loss

        return _compute_reg_loss

    def post_task_cleanup(
        self,
        task_idx: int,
        trainer: Any,
    ) -> None:
        """Update snapshots after task completion."""
        logger.info(
            "[CLoRAScaffold] Task %d complete. Regularization was %s. "
            "Importance entries: %d.",
            task_idx,
            "active" if self._active else "inactive",
            len(self._importance),
        )
