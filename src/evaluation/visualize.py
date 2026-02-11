"""Visualization utilities for VecAdvisor++ evaluation results.

Generates plots comparing baseline vs advisor configurations.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from src.benchmark.metrics import BenchmarkResult


def plot_recall_vs_latency(
    results: list[BenchmarkResult],
    output_dir: str = "results/plots",
    filename: str = "recall_vs_latency.png",
) -> str:
    """Plot recall@k vs p95 latency for all configurations.

    Args:
        results: List of BenchmarkResults.
        output_dir: Directory to save the plot.
        filename: Output filename.

    Returns:
        Path to saved plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for r in results:
        marker = "s" if r.config_name == "vecadvisor++" else "o"
        size = 150 if r.config_name == "vecadvisor++" else 80
        color = "red" if r.config_name == "vecadvisor++" else None
        ax.scatter(
            r.latency_p95_ms, r.recall,
            s=size, marker=marker, color=color,
            label=r.config_name, zorder=5,
        )

    ax.set_xlabel("p95 Latency (ms)", fontsize=12)
    ax.set_ylabel("Recall@k", fontsize=12)
    ax.set_title("Recall vs Latency Tradeoff", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_build_time_comparison(
    results: list[BenchmarkResult],
    output_dir: str = "results/plots",
    filename: str = "build_time.png",
) -> str:
    """Bar chart of index build times.

    Args:
        results: List of BenchmarkResults.
        output_dir: Directory to save the plot.
        filename: Output filename.

    Returns:
        Path to saved plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    names = [r.config_name for r in results]
    build_times = [r.build_time_s for r in results]
    colors = ["red" if n == "vecadvisor++" else "steelblue" for n in names]

    bars = ax.bar(names, build_times, color=colors)
    ax.set_ylabel("Build Time (seconds)", fontsize=12)
    ax.set_title("Index Build Time Comparison", fontsize=14)
    ax.tick_params(axis="x", rotation=30)

    for bar, val in zip(bars, build_times):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}s", ha="center", va="bottom", fontsize=9,
        )

    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_completion_rate(
    results: list[BenchmarkResult],
    output_dir: str = "results/plots",
    filename: str = "completion_rate.png",
) -> str:
    """Bar chart of top-k completion rates (critical for filtered queries).

    Args:
        results: List of BenchmarkResults.
        output_dir: Directory to save the plot.
        filename: Output filename.

    Returns:
        Path to saved plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    names = [r.config_name for r in results]
    rates = [r.completion_rate * 100 for r in results]
    colors = ["red" if n == "vecadvisor++" else "steelblue" for n in names]

    bars = ax.bar(names, rates, color=colors)
    ax.set_ylabel("Completion Rate (%)", fontsize=12)
    ax.set_title("Top-k Completion Rate (Filtered Queries)", fontsize=14)
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=30)

    for bar, val in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9,
        )

    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_selectivity_heatmap(
    results_by_selectivity: dict[str, list[BenchmarkResult]],
    metric: str = "recall",
    output_dir: str = "results/plots",
    filename: str = "selectivity_heatmap.png",
) -> str:
    """Heatmap of a metric across selectivity levels and configurations.

    Args:
        results_by_selectivity: Dict mapping selectivity label to BenchmarkResults.
        metric: Metric to plot ("recall", "latency_p95_ms", "completion_rate").
        output_dir: Directory to save.
        filename: Output filename.

    Returns:
        Path to saved plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    selectivities = list(results_by_selectivity.keys())
    if not selectivities:
        return ""

    config_names = [r.config_name for r in results_by_selectivity[selectivities[0]]]

    data = np.zeros((len(config_names), len(selectivities)))
    for j, sel in enumerate(selectivities):
        for i, r in enumerate(results_by_selectivity[sel]):
            data[i, j] = getattr(r, metric)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(selectivities)))
    ax.set_xticklabels(selectivities, fontsize=10)
    ax.set_yticks(range(len(config_names)))
    ax.set_yticklabels(config_names, fontsize=10)
    ax.set_xlabel("Filter Selectivity", fontsize=12)
    ax.set_title(f"{metric} across Selectivity Levels", fontsize=14)

    # Add value annotations
    for i in range(len(config_names)):
        for j in range(len(selectivities)):
            ax.text(
                j, i, f"{data[i, j]:.3f}",
                ha="center", va="center", fontsize=8,
            )

    fig.colorbar(im, ax=ax)

    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_all_plots(
    results: list[BenchmarkResult],
    output_dir: str = "results/plots",
) -> list[str]:
    """Generate all standard comparison plots.

    Args:
        results: List of BenchmarkResults.
        output_dir: Output directory.

    Returns:
        List of paths to generated plots.
    """
    paths = [
        plot_recall_vs_latency(results, output_dir),
        plot_build_time_comparison(results, output_dir),
        plot_completion_rate(results, output_dir),
    ]
    return [p for p in paths if p]
