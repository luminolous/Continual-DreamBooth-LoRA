"""
Faithful C-LoRA continual learning method with occupancy regularization.

This method implements a shared continual model where each task gets its own
LoRA adapter. All adapters are composed (summed) into a single effective delta
at inference time. Task identity comes from personalized token embeddings, not
adapter selection.

== Core mechanism ==

1. Per-task adapters: each task creates a fresh LoRA adapter (PEFT add_adapter).
   Previous adapters are frozen but remain active — their deltas are always
   summed into the continual model.

2. Per-task tokens: each task registers a randomly initialized token embedding.
   Only that token's embedding row is trained.

3. Occupancy regularization: prevents new adapter deltas from interfering with
   past adapter deltas in the composed model.

   For each past task k, we store the LoRA factor matrices (A_k, B_k).
   We compute an occupancy mask: O^m = normalize(Σ_k |B_k @ A_k|)
   The new adapter is penalized for placing large deltas in occupied positions:
     L_occ = λ * Σ_m ||O^m ⊙ (B_new @ A_new)||²

   This is an *interference/occupancy* constraint, not a distance-to-old-weights
   penalty. The new adapter is free in unoccupied weight space.

== Faithfulness ==

This is an implementation approximation of C-LoRA's core idea:
- Exact: per-task adapters, token embeddings, cross-attention K/V targeting,
  past LoRA factors inform regularization
- Approximate: occupancy mask is magnitude-based (|B@A|), not gradient/subspace-aware
- See implementation_plan.md for full faithfulness assessment
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from methods.base import BaseMethod

logger = logging.getLogger(__name__)


class FaithfulCLoRA(BaseMethod):
    """Faithful C-LoRA: shared continual model with occupancy regularization.

    Primary evaluation uses the composed model (all adapters active).
    Forgetting occurs when new adapter deltas interfere with past adapter
    contributions in the composed model. The occupancy regularizer prevents this.

    Lifecycle per task:
        1. pre_task_setup: archive past factors, build occupancy masks
        2. setup_adapters: create new adapter, freeze old, register token
        3. get_extra_loss_fn: return occupancy regularization closure
        4. <training>
        5. post_task_cleanup: snapshot completed adapter factors, update occupancy
    """

    def __init__(
        self,
        regularization_weight: float = 0.1,
        token_init: str = "random",
    ):
        """
        Args:
            regularization_weight: Global occupancy regularization strength (λ).
            token_init: Token initialization mode ("random" or "fixed").
        """
        self.regularization_weight = regularization_weight
        self.token_init = token_init

        # Storage for past adapter LoRA factors
        # _past_factors[k] = {module_name: (A_k, B_k)} for completed task k
        self._past_factors: List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = []

        # Accumulated occupancy masks per module
        # _occupancy[module_name] = Tensor of shape (out_features, in_features)
        self._occupancy: Dict[str, torch.Tensor] = {}

        # Current task's adapter name (set during setup_adapters)
        self._current_adapter: Optional[str] = None

        # Whether regularization is active (not for task 0)
        self._active = False

    def pre_task_setup(
        self,
        task_idx: int,
        trainer: Any,
        config: Any,
    ) -> None:
        """Archive past adapter factors and build occupancy masks.

        For task 0: no regularization active.
        For task t>0: occupancy masks already built from post_task_cleanup
        of the previous task.
        """
        if task_idx == 0:
            self._active = False
            logger.info(
                "[FaithfulCLoRA] Task 0: first task, no regularization active"
            )
        else:
            self._active = True
            logger.info(
                "[FaithfulCLoRA] Task %d: occupancy regularization active "
                "(λ=%.4f, %d past adapters, %d occupancy masks)",
                task_idx,
                self.regularization_weight,
                len(self._past_factors),
                len(self._occupancy),
            )

    def setup_adapters(
        self,
        task_idx: int,
        trainer: Any,
        config: Any,
    ) -> None:
        task = config.tasks[task_idx]
        adapter_name = f"task_{task_idx}"

        token_id = trainer.register_task_token(
            task.trigger_token,
            init_mode=self.token_init,
        )

        trainer.create_task_adapter(adapter_name)
        self._current_adapter = adapter_name

        if config.c_lora.train_token_embeddings:
            trainer.set_token_embeddings_trainable([token_id])

        logger.info(
            "[FaithfulCLoRA] Task %d: new adapter '%s', token '<%s>' (ID=%d)",
            task_idx, adapter_name, task.trigger_token, token_id,
        )

    def get_extra_loss_fn(self) -> Optional[Callable[[torch.nn.Module], torch.Tensor]]:
        """Return occupancy regularization loss function.

        Penalizes the new adapter's effective delta (B_new @ A_new) in positions
        where past adapters had significant updates (high occupancy).

        Returns None for task 0 (no past information to regularize against).
        """
        if not self._active or not self._occupancy:
            return None

        occupancy = self._occupancy
        reg_weight = self.regularization_weight
        current_adapter = self._current_adapter

        def _compute_occupancy_loss(unet: torch.nn.Module) -> torch.Tensor:
            device = next(unet.parameters()).device
            total_loss = torch.tensor(0.0, device=device)
            matched = 0

            # Collect current adapter's A and B matrices
            current_factors: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
            params_dict = dict(unet.named_parameters())

            for name, param in params_dict.items():
                if current_adapter not in name:
                    continue
                if "lora_A" in name and param.requires_grad:
                    module_key = name.split(".lora_A.")[0]
                    a_weight = param  # keep on device, has grad

                    b_name = name.replace("lora_A", "lora_B")
                    b_param = params_dict.get(b_name)
                    if b_param is not None:
                        current_factors[module_key] = (a_weight, b_param)

            for module_key, (a_mat, b_mat) in current_factors.items():
                if module_key not in occupancy:
                    continue

                # Compute effective delta: B @ A
                # A shape: (rank, in_features), B shape: (out_features, rank)
                delta_new = torch.matmul(b_mat, a_mat)  # (out, in)

                # Get occupancy mask for this module
                occ_mask = occupancy[module_key].to(device)

                # Penalize delta in occupied positions
                interference = occ_mask * delta_new
                module_loss = torch.sum(interference ** 2)
                total_loss = total_loss + module_loss
                matched += 1

            if matched == 0:
                logger.warning(
                    "[FaithfulCLoRA] No modules matched for occupancy loss. "
                    "Check adapter name '%s' and occupancy keys.", current_adapter,
                )
                return torch.tensor(0.0, device=device)

            return reg_weight * total_loss

        return _compute_occupancy_loss

    def post_task_cleanup(
        self,
        task_idx: int,
        trainer: Any,
    ) -> None:
        """Archive the completed adapter's factors and update occupancy.

        After training task t:
        1. Extract (A_t, B_t) factor matrices from the just-trained adapter
        2. Compute ΔW_t = B_t @ A_t for each module
        3. Accumulate |ΔW_t| into the occupancy mask
        4. Store factors for future reference
        """
        adapter_name = f"task_{task_idx}"

        # Extract factor matrices
        factors = trainer.get_adapter_lora_factors(adapter_name)
        self._past_factors.append(factors)

        # Update occupancy masks
        for module_key, (a_mat, b_mat) in factors.items():
            # Compute effective delta on CPU
            delta = torch.matmul(b_mat, a_mat)  # (out, in)
            abs_delta = delta.abs()

            if module_key in self._occupancy:
                self._occupancy[module_key] = self._occupancy[module_key] + abs_delta
            else:
                self._occupancy[module_key] = abs_delta

        # Normalize occupancy masks to [0, 1] range
        for module_key in self._occupancy:
            occ = self._occupancy[module_key]
            max_val = occ.max()
            if max_val > 0:
                self._occupancy[module_key] = occ / max_val

        logger.info(
            "[FaithfulCLoRA] Task %d complete. Archived %d factor pairs. "
            "Occupancy masks: %d modules. Past adapters: %d.",
            task_idx,
            len(factors),
            len(self._occupancy),
            len(self._past_factors),
        )

def export_state(self) -> dict:
    return {
        "past_factors": self._past_factors,
        "occupancy": self._occupancy,
        "current_adapter": self._current_adapter,
        "active": self._active,
    }

def load_state(self, state: dict) -> None:
    self._past_factors = state.get("past_factors", [])
    self._occupancy = state.get("occupancy", {})
    self._current_adapter = state.get("current_adapter")
    self._active = state.get("active", False)