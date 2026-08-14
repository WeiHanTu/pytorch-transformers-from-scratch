"""Regenerate README result figures from the tracked benchmark JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.container import BarContainer


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "results" / "benchmark.json"
OUTPUT_DIRECTORY = ROOT / "docs" / "figures"
BLUE = "#1769AA"
TEAL = "#00897B"
SLATE = "#607D8B"
LIGHT_SLATE = "#B0BEC5"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def label_horizontal_bars(axis: plt.Axes, bars: BarContainer, suffix: str = "") -> None:
    for bar in bars:
        value = bar.get_width()
        axis.text(
            value + 0.7,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}{suffix}",
            va="center",
            fontsize=9,
            weight="semibold",
        )


def classification_figure(benchmark: dict[str, Any]) -> Path:
    variants = benchmark["classification"]["variants"]
    keys = ["absolute", "alibi", "local", "disentangled"]
    labels = ["Absolute", "ALiBi", "Local (±4)", "Disentangled"]
    train = [variants[key]["train_accuracy"] * 100 for key in keys]
    test = [variants[key]["test_accuracy"] * 100 for key in keys]
    positions = list(range(len(keys)))

    figure, axis = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    train_bars = axis.barh(
        [position + 0.18 for position in positions],
        train,
        height=0.32,
        color=LIGHT_SLATE,
        label="Train",
    )
    test_bars = axis.barh(
        [position - 0.18 for position in positions],
        test,
        height=0.32,
        color=BLUE,
        label="Test",
    )
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 105)
    axis.set_xlabel("Accuracy (%)")
    axis.set_title(
        "Speech classification: controlled 15-epoch comparison",
        loc="left",
        pad=28,
    )
    axis.text(
        0,
        1.012,
        "Same tokenizer, optimizer, seed, and training budget across all variants",
        transform=axis.transAxes,
        color=SLATE,
        fontsize=9,
    )
    axis.xaxis.grid(True, color="#E8EDF2", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, ncol=2, loc="lower right")
    label_horizontal_bars(axis, train_bars, "%")
    label_horizontal_bars(axis, test_bars, "%")

    destination = OUTPUT_DIRECTORY / "classification-results.png"
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination


def language_model_figure(benchmark: dict[str, Any]) -> Path:
    language_model = benchmark["language_model"]
    history = language_model["history"]
    steps = [point["step"] for point in history]
    train_perplexity = [point["train_perplexity"] for point in history]
    test_names = ["Obama", "G. W. Bush", "G. H. W. Bush"]
    test_values = [
        language_model["test_perplexity"]["obama"],
        language_model["test_perplexity"]["g_w_bush"],
        language_model["test_perplexity"]["ghw_bush"],
    ]

    figure, (curve_axis, bars_axis) = plt.subplots(
        1, 2, figsize=(11, 4.7), constrained_layout=True, gridspec_kw={"width_ratios": [1.3, 1]}
    )
    curve_axis.plot(steps, train_perplexity, color=TEAL, marker="o", linewidth=2.3)
    for step, value in zip(steps, train_perplexity):
        curve_axis.annotate(
            f"{value:.0f}",
            (step, value),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            weight="semibold",
        )
    curve_axis.set_title("Training convergence", loc="left")
    curve_axis.set_xlabel("Optimizer step")
    curve_axis.set_ylabel("Perplexity (lower is better)")
    curve_axis.set_xticks(steps)
    curve_axis.set_ylim(0, 560)
    curve_axis.yaxis.grid(True, color="#E8EDF2", linewidth=0.8)
    curve_axis.set_axisbelow(True)

    bars = bars_axis.barh(test_names, test_values, color=[BLUE, SLATE, TEAL], height=0.55)
    bars_axis.invert_yaxis()
    bars_axis.set_xlim(0, 525)
    bars_axis.set_xlabel("Perplexity (lower is better)")
    bars_axis.set_title("Held-out speech splits", loc="left")
    bars_axis.xaxis.grid(True, color="#E8EDF2", linewidth=0.8)
    bars_axis.set_axisbelow(True)
    label_horizontal_bars(bars_axis, bars)
    figure.suptitle(
        "Decoder-only language modeling · 500 training steps",
        fontsize=14,
        weight="bold",
    )

    destination = OUTPUT_DIRECTORY / "language-model-results.png"
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination


def main() -> None:
    configure_style()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    for destination in (classification_figure(benchmark), language_model_figure(benchmark)):
        print(destination.relative_to(ROOT))


if __name__ == "__main__":
    main()
