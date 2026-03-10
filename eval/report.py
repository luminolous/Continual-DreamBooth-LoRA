"""
Report generation: CSV/JSON export, score matrix heatmap, line charts,
forgetting bar chart, and text summary.

Phase 2 additions:
- Per-task score progression line chart
- Forgetting bar chart
- Confusion gap heatmap (when enabled)
- Per-prompt breakdown table
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from utils.io import ensure_dir, save_json

logger = logging.getLogger(__name__)


def export_score_matrix_csv(
    score_matrix: np.ndarray,
    task_names: List[str],
    output_path: str,
) -> None:
    """Export the score matrix as a CSV file.

    Rows = training stages, Columns = evaluated tasks.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["stage"] + task_names
        writer.writerow(header)

        for t in range(score_matrix.shape[0]):
            row = [f"after_task_{t}"]
            for j in range(score_matrix.shape[1]):
                val = score_matrix[t][j]
                row.append(f"{val:.4f}" if not np.isnan(val) else "")
            writer.writerow(row)

    logger.info("Score matrix CSV saved to %s", path)


def export_metrics_json(
    metrics: Dict[str, Any],
    task_names: List[str],
    output_path: str,
    extra_data: Optional[Dict[str, Any]] = None,
) -> None:
    """Export forgetting metrics and summary as JSON."""
    enriched = {
        "task_names": task_names,
        **metrics,
    }
    if extra_data:
        enriched.update(extra_data)
    save_json(enriched, output_path)
    logger.info("Metrics JSON saved to %s", output_path)


# ---------------------------------------------------------------------------
# Matplotlib plots
# ---------------------------------------------------------------------------

def _get_plt():
    """Import matplotlib with Agg backend. Returns None if unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot generation")
        return None


def generate_heatmap(
    score_matrix: np.ndarray,
    task_names: List[str],
    output_path: str,
    title: str = "CCIP Score Matrix",
) -> None:
    """Generate a heatmap plot of the score matrix."""
    plt = _get_plt()
    if plt is None:
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    T = score_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(max(6, T * 1.5), max(5, T * 1.2)))

    masked = np.ma.masked_invalid(score_matrix)
    cmap = plt.cm.RdYlGn
    cmap.set_bad(color="lightgray")

    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="CCIP Score")

    ax.set_xticks(range(T))
    ax.set_xticklabels(task_names, rotation=45, ha="right")
    ax.set_yticks(range(T))
    ax.set_yticklabels([f"After task {t}" for t in range(T)])
    ax.set_xlabel("Evaluated Task")
    ax.set_ylabel("Training Stage")
    ax.set_title(title)

    for t in range(T):
        for j in range(T):
            val = score_matrix[t][j]
            if not np.isnan(val):
                text_color = "white" if val < 0.4 or val > 0.8 else "black"
                ax.text(
                    j, t, f"{val:.2f}",
                    ha="center", va="center",
                    color=text_color, fontsize=9,
                )

    plt.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Heatmap saved to %s", path)


def generate_score_progression_chart(
    score_matrix: np.ndarray,
    task_names: List[str],
    output_path: str,
    title: str = "Per-Task Score Progression",
) -> None:
    """Generate a line chart showing how each task's score evolves over stages.

    Each line represents one task, plotted across training stages.
    The chart shows when forgetting occurs (score drops) and how severe it is.
    """
    plt = _get_plt()
    if plt is None:
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    T = score_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(max(8, T * 1.5), 5))

    colors = plt.cm.tab10(np.linspace(0, 1, T))  # Distinct colors per task

    for j in range(T):
        scores = score_matrix[:, j]
        # Only plot from the stage when this task was first seen
        stages = list(range(j, T))
        valid_scores = scores[j:]

        ax.plot(
            stages,
            valid_scores,
            marker="o",
            label=task_names[j],
            color=colors[j],
            linewidth=2,
            markersize=6,
        )

        # Mark the point where the task was just learned (diagonal)
        if not np.isnan(scores[j]):
            ax.scatter(
                [j], [scores[j]],
                color=colors[j], s=120, zorder=5,
                edgecolors="black", linewidths=1.5,
            )

    ax.set_xlabel("Training Stage (after task N)")
    ax.set_ylabel("CCIP Score")
    ax.set_title(title)
    ax.set_xticks(range(T))
    ax.set_xticklabels([f"After {n}" for n in task_names], rotation=30, ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Score progression chart saved to %s", path)


def generate_forgetting_bar_chart(
    metrics: Dict[str, Any],
    task_names: List[str],
    output_path: str,
    title: str = "Per-Task Forgetting",
) -> None:
    """Generate a bar chart showing forgetting per task.

    Also shows backward transfer as a separate color.
    """
    plt = _get_plt()
    if plt is None:
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    forgetting = metrics["per_task_forgetting"]
    T = len(forgetting)

    fig, ax = plt.subplots(figsize=(max(6, T * 1.2), 5))

    x = np.arange(T)
    bars = ax.bar(x, forgetting, color="coral", alpha=0.85, edgecolor="black", linewidth=0.5)

    # Color the last bar differently (no forgetting by definition)
    if T > 1:
        bars[-1].set_color("lightgray")
        bars[-1].set_alpha(0.5)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, forgetting)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    # Add average line
    avg_forg = metrics.get("average_forgetting", 0)
    ax.axhline(y=avg_forg, color="red", linestyle="--", alpha=0.7,
               label=f"Avg forgetting: {avg_forg:.3f}")

    ax.set_xlabel("Task")
    ax.set_ylabel("Forgetting (best - final)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(task_names, rotation=30, ha="right")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Forgetting bar chart saved to %s", path)


def generate_confusion_gap_heatmap(
    confusion_data: Dict[int, Dict[int, Dict[str, Any]]],
    task_names: List[str],
    output_path: str,
    title: str = "Confusion Gap Matrix",
) -> None:
    """Generate a heatmap of confusion gaps.

    Args:
        confusion_data: confusion_data[stage_t][task_j] = confusion gap dict
        task_names: List of task names.
        output_path: Path to save the PNG plot.
        title: Plot title.
    """
    plt = _get_plt()
    if plt is None:
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    T = len(task_names)
    gap_matrix = np.full((T, T), np.nan)

    for t, stage_data in confusion_data.items():
        for j, gap_info in stage_data.items():
            gap_matrix[int(t)][int(j)] = gap_info.get("confusion_gap", np.nan)

    fig, ax = plt.subplots(figsize=(max(6, T * 1.5), max(5, T * 1.2)))

    masked = np.ma.masked_invalid(gap_matrix)
    # Diverging colormap: negative gap (confusion) = red, positive = blue
    cmap = plt.cm.RdBu
    cmap.set_bad(color="lightgray")

    vmax = max(abs(np.nanmin(gap_matrix)) if not np.all(np.isnan(gap_matrix)) else 1,
               abs(np.nanmax(gap_matrix)) if not np.all(np.isnan(gap_matrix)) else 1)
    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Confusion Gap")

    ax.set_xticks(range(T))
    ax.set_xticklabels(task_names, rotation=45, ha="right")
    ax.set_yticks(range(T))
    ax.set_yticklabels([f"After task {t}" for t in range(T)])
    ax.set_xlabel("Evaluated Task")
    ax.set_ylabel("Training Stage")
    ax.set_title(title)

    for t in range(T):
        for j in range(T):
            val = gap_matrix[t][j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(
                    j, t, f"{val:.2f}",
                    ha="center", va="center",
                    color=text_color, fontsize=9,
                )

    plt.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion gap heatmap saved to %s", path)


def generate_summary_text(
    metrics: Dict[str, Any],
    task_names: List[str],
) -> str:
    """Generate a human-readable text summary of the evaluation results."""
    lines = [
        "=" * 60,
        "CONTINUAL DREAMBOOTH-CLORA EVALUATION SUMMARY",
        "=" * 60,
        "",
        f"Final Average Score:       {metrics['final_average_score']:.4f}",
        f"Average Forgetting:       {metrics['average_forgetting']:.4f}",
        f"Avg Backward Transfer:    {metrics.get('average_backward_transfer', 0):.4f}",
        "",
        "Per-Stage Average Scores:",
    ]

    for t, avg in enumerate(metrics["per_stage_average_score"]):
        seen = ", ".join(task_names[: t + 1])
        lines.append(f"  After task {t} ({task_names[t]}): {avg:.4f}  [seen: {seen}]")

    # Learning accuracy (diagonal)
    if "learning_accuracy" in metrics:
        lines.append("")
        lines.append("Learning Accuracy (diagonal A[t][t]):")
        for j, acc in enumerate(metrics["learning_accuracy"]):
            lines.append(f"  {task_names[j]}: {acc:.4f}")

    lines.append("")
    lines.append("Per-Task Forgetting:")
    for j, forg in enumerate(metrics["per_task_forgetting"]):
        lines.append(f"  {task_names[j]}: {forg:.4f}")

    # Backward transfer
    if "backward_transfer" in metrics and metrics["backward_transfer"]:
        lines.append("")
        lines.append("Backward Transfer (final - immediate, negative = forgetting):")
        for j, bt in enumerate(metrics["backward_transfer"]):
            lines.append(f"  {task_names[j]}: {bt:+.4f}")

    lines.append("")
    lines.append("Score Matrix:")
    matrix = metrics["score_matrix"]
    header = "          " + "  ".join(f"{name:>10}" for name in task_names)
    lines.append(header)
    for t, row in enumerate(matrix):
        vals = "  ".join(
            f"{v:>10.4f}" if v is not None and not np.isnan(v) else f"{'':>10}"
            for v in row
        )
        lines.append(f"Stage {t}:  {vals}")

    lines.append("=" * 60)
    return "\n".join(lines)


def save_full_report(
    score_matrix: np.ndarray,
    metrics: Dict[str, Any],
    task_names: List[str],
    output_dir: str,
    experiment_name: str = "experiment",
    confusion_data: Optional[Dict] = None,
    per_prompt_data: Optional[Dict] = None,
) -> None:
    """Generate and save all report artifacts.

    Saves:
    - score_matrix.csv
    - metrics.json
    - heatmap.png
    - score_progression.png (Phase 2)
    - forgetting_bar.png (Phase 2)
    - confusion_gap_heatmap.png (Phase 2, if confusion data provided)
    - summary.txt

    Args:
        score_matrix: The T×T CCIP score matrix.
        metrics: Forgetting metrics dict.
        task_names: List of task names.
        output_dir: Base output directory.
        experiment_name: Name for the report title.
        confusion_data: Optional confusion gap data.
        per_prompt_data: Optional per-prompt breakdown data.
    """
    report_dir = ensure_dir(Path(output_dir) / "reports")

    # --- Core artifacts ---
    export_score_matrix_csv(
        score_matrix, task_names, str(report_dir / "score_matrix.csv")
    )

    extra_json = {}
    if confusion_data:
        extra_json["confusion_gap_data"] = confusion_data
    if per_prompt_data:
        extra_json["per_prompt_scores"] = per_prompt_data

    export_metrics_json(
        metrics, task_names, str(report_dir / "metrics.json"),
        extra_data=extra_json if extra_json else None,
    )

    # --- Plots ---
    generate_heatmap(
        score_matrix, task_names, str(report_dir / "heatmap.png"),
        title=f"CCIP Score Matrix — {experiment_name}",
    )

    generate_score_progression_chart(
        score_matrix, task_names, str(report_dir / "score_progression.png"),
        title=f"Score Progression — {experiment_name}",
    )

    generate_forgetting_bar_chart(
        metrics, task_names, str(report_dir / "forgetting_bar.png"),
        title=f"Per-Task Forgetting — {experiment_name}",
    )

    if confusion_data:
        generate_confusion_gap_heatmap(
            confusion_data, task_names,
            str(report_dir / "confusion_gap_heatmap.png"),
            title=f"Confusion Gap — {experiment_name}",
        )

    # --- Text summary ---
    summary = generate_summary_text(metrics, task_names)
    summary_path = report_dir / "summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    logger.info("Summary saved to %s", summary_path)

    # Also print summary to log
    logger.info("\n%s", summary)
