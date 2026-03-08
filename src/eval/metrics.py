"""
CCIP-based identity scoring, confusion gap, and forgetting matrix computation.

Uses sdeval (pinned: 0.2.4) for anime character identity evaluation.
The CCIPMetrics class computes character identity similarity between
generated images and reference images.

Metric definitions:
- Score matrix A[t][j]: CCIP score for task j after training task t
- Per-task score: diagonal A[t][t]
- Average seen-task score after task t: mean(A[t][0..t])
- Final average score: mean(A[T-1][0..T-1])
- Forgetting for task j: max(A[t][j] for t in 0..T-1) - A[T-1][j]
- Average forgetting: mean of per-task forgetting (excluding last task)

Phase 2 additions:
- Confusion gap: target_score - max(non_target_scores)  (higher = better discrimination)
- Per-prompt breakdown: CCIP scores computed per evaluation prompt template
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_ccip_score(
    generated_dir: str,
    ref_dir: str,
) -> float:
    """Compute CCIP identity score between generated and reference images.

    Uses sdeval.fidelity.CCIPMetrics to evaluate how well generated images
    preserve the identity of the reference character.

    Args:
        generated_dir: Directory containing generated evaluation images.
        ref_dir: Directory containing reference images for the character.

    Returns:
        CCIP similarity score (higher = better identity preservation).
        Returns 0.0 if evaluation fails.
    """
    try:
        from sdeval.fidelity import CCIPMetrics
    except ImportError:
        logger.error(
            "sdeval is not installed. Install with: pip install sdeval==0.2.4\n"
            "CCIP evaluation requires this exact version for API compatibility."
        )
        return 0.0

    try:
        # CCIPMetrics takes reference images as the "true" distribution
        ccip = CCIPMetrics(images=ref_dir)
        score = ccip.score(generated_dir)

        # score can be a dict or float depending on sdeval version
        if isinstance(score, dict):
            # Some versions return {"ccip": float_value}
            score = score.get("ccip", score.get("score", 0.0))

        score = float(score)
        logger.info(
            "CCIP score: %.4f (generated=%s, ref=%s)",
            score, generated_dir, ref_dir,
        )
        return score

    except Exception as e:
        logger.error("CCIP evaluation failed: %s", e, exc_info=True)
        return 0.0


def compute_per_prompt_scores(
    generated_dir: str,
    ref_dir: str,
    num_prompts: int,
    num_images_per_prompt: int,
    task_name: str,
) -> Dict[str, float]:
    """Compute CCIP scores broken down by prompt template.

    Assumes generated images follow the naming convention:
        {task_name}_p{prompt_idx:02d}_i{img_idx:02d}.png

    Args:
        generated_dir: Directory with generated images.
        ref_dir: Reference images directory.
        num_prompts: Number of prompt templates used.
        num_images_per_prompt: Images generated per prompt.
        task_name: Task name for filename matching.

    Returns:
        Dict mapping prompt index string to CCIP score.
    """
    import tempfile
    import shutil

    gen_path = Path(generated_dir)
    results = {}

    for p_idx in range(num_prompts):
        # Collect images for this prompt into a temp directory
        prompt_images = sorted(
            gen_path.glob(f"{task_name}_p{p_idx:02d}_i*.png")
        )

        if not prompt_images:
            results[f"prompt_{p_idx:02d}"] = 0.0
            continue

        with tempfile.TemporaryDirectory() as tmp_dir:
            for img in prompt_images:
                shutil.copy2(str(img), tmp_dir)

            score = compute_ccip_score(tmp_dir, ref_dir)
            results[f"prompt_{p_idx:02d}"] = score

    return results


def compute_confusion_gap(
    generated_dir: str,
    target_ref_dir: str,
    other_ref_dirs: List[Tuple[str, str]],
) -> Dict[str, Any]:
    """Compute confusion gap: target score minus max non-target score.

    A positive confusion gap means the model generates images closer to the
    target character than to any other character. A negative gap indicates
    identity confusion.

    Args:
        generated_dir: Directory with generated images for the target character.
        target_ref_dir: Reference images for the target character.
        other_ref_dirs: List of (task_name, ref_dir) for non-target characters.

    Returns:
        Dictionary with:
        - target_score: CCIP score against target references
        - other_scores: dict of {task_name: score} for non-targets
        - max_non_target_score: highest score among non-targets
        - confusion_gap: target_score - max_non_target_score
    """
    target_score = compute_ccip_score(generated_dir, target_ref_dir)

    other_scores = {}
    for name, ref_dir in other_ref_dirs:
        other_scores[name] = compute_ccip_score(generated_dir, ref_dir)

    max_non_target = max(other_scores.values()) if other_scores else 0.0
    gap = target_score - max_non_target

    logger.info(
        "Confusion gap: %.4f (target=%.4f, max_other=%.4f)",
        gap, target_score, max_non_target,
    )

    return {
        "target_score": target_score,
        "other_scores": other_scores,
        "max_non_target_score": max_non_target,
        "confusion_gap": gap,
    }


def build_score_matrix(
    scores: Dict[int, Dict[int, float]],
    num_tasks: int,
) -> np.ndarray:
    """Build the task-by-task score matrix.

    Args:
        scores: Nested dict scores[stage_t][task_j] = ccip_score.
        num_tasks: Total number of tasks.

    Returns:
        numpy array of shape (num_tasks, num_tasks) where A[t][j] is the
        CCIP score for task j after completing stage t.
        Entries for unseen tasks (j > t) are NaN.
    """
    matrix = np.full((num_tasks, num_tasks), np.nan)

    for t, task_scores in scores.items():
        for j, score in task_scores.items():
            matrix[t][j] = score

    return matrix


def compute_forgetting_metrics(score_matrix: np.ndarray) -> Dict[str, Any]:
    """Compute forgetting and summary metrics from the score matrix.

    Args:
        score_matrix: Array of shape (T, T) where A[t][j] is the CCIP
                      score for task j after completing stage t.

    Returns:
        Dictionary with:
        - per_task_forgetting: list of per-task forgetting values
        - average_forgetting: mean forgetting across tasks (excl. last)
        - final_average_score: mean score after all tasks
        - per_stage_average: list of average seen-task scores per stage
        - backward_transfer: average change in score for old tasks after learning new
        - learning_accuracy: diagonal scores A[t][t] (score right after learning)
    """
    T = score_matrix.shape[0]

    # Per-task forgetting: best score ever achieved minus final score
    per_task_forgetting = []
    for j in range(T):
        valid_scores = score_matrix[:, j]
        valid_scores = valid_scores[~np.isnan(valid_scores)]
        if len(valid_scores) > 0:
            best = np.max(valid_scores)
            final = valid_scores[-1]  # Score after last stage
            per_task_forgetting.append(float(best - final))
        else:
            per_task_forgetting.append(0.0)

    # Average forgetting (exclude the last task — it has no forgetting by definition)
    if T > 1:
        avg_forgetting = float(np.mean(per_task_forgetting[:-1]))
    else:
        avg_forgetting = 0.0

    # Final average score (after all tasks are trained)
    final_row = score_matrix[-1]
    valid_final = final_row[~np.isnan(final_row)]
    final_avg = float(np.mean(valid_final)) if len(valid_final) > 0 else 0.0

    # Per-stage average of seen tasks
    per_stage_avg = []
    for t in range(T):
        seen_scores = score_matrix[t, : t + 1]
        valid = seen_scores[~np.isnan(seen_scores)]
        avg = float(np.mean(valid)) if len(valid) > 0 else 0.0
        per_stage_avg.append(avg)

    # Learning accuracy: diagonal entries A[t][t]
    learning_acc = [float(score_matrix[t][t]) for t in range(T)]

    # Backward transfer: how much old task scores drop after learning new ones
    backward_transfer = []
    for j in range(T - 1):
        # Score right after learning task j vs. final score
        immediate = score_matrix[j][j]
        final = score_matrix[-1][j]
        if not np.isnan(immediate) and not np.isnan(final):
            backward_transfer.append(float(final - immediate))
        else:
            backward_transfer.append(0.0)
    avg_backward_transfer = (
        float(np.mean(backward_transfer)) if backward_transfer else 0.0
    )

    return {
        "per_task_forgetting": per_task_forgetting,
        "average_forgetting": avg_forgetting,
        "final_average_score": final_avg,
        "per_stage_average_score": per_stage_avg,
        "learning_accuracy": learning_acc,
        "backward_transfer": backward_transfer,
        "average_backward_transfer": avg_backward_transfer,
        "score_matrix": score_matrix.tolist(),
    }
