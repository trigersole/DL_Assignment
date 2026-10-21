from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_metrics(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def plot_training_curves(metrics: list[dict[str, float]], output_path: Path, title: str) -> None:
    epochs = [int(row["epoch"]) for row in metrics]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, [row["train_loss"] for row in metrics], label="Training")
    axes[0].plot(epochs, [row["validation_loss"] for row in metrics], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss")
    axes[1].plot(epochs, [100 * row["train_accuracy"] for row in metrics], label="Training")
    axes[1].plot(epochs, [100 * row["validation_accuracy"] for row in metrics], label="Validation")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Accuracy")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(title)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_experiment_comparison(
    experiments: list[tuple[str, list[dict[str, float]]]], output_path: Path, title: str
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = [
        ("train_loss", "Training loss", "Cross-entropy loss", 1.0),
        ("validation_loss", "Validation loss", "Cross-entropy loss", 1.0),
        ("train_accuracy", "Training accuracy", "Accuracy (%)", 100.0),
        ("validation_accuracy", "Validation accuracy", "Accuracy (%)", 100.0),
    ]
    for axis, (key, panel_title, ylabel, scale) in zip(axes.flat, panels):
        for label, metrics in experiments:
            axis.plot(
                [int(row["epoch"]) for row in metrics],
                [scale * row[key] for row in metrics],
                label=label,
            )
        axis.set(xlabel="Epoch", ylabel=ylabel, title=panel_title)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(title)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_beta_density(alpha: float, output_path: Path) -> None:
    from math import gamma

    x = np.linspace(0.001, 0.999, 1000)
    coefficient = gamma(2 * alpha) / (gamma(alpha) ** 2)
    density = coefficient * x ** (alpha - 1) * (1 - x) ** (alpha - 1)
    figure, axis = plt.subplots(figsize=(6.2, 4.2))
    axis.plot(x, density, color="#276FBF", linewidth=2)
    axis.fill_between(x, density, alpha=0.18, color="#276FBF")
    axis.set(
        xlabel=r"Mixing coefficient $\lambda$",
        ylabel="Probability density",
        title=rf"Beta distribution, $\alpha=\beta={alpha:g}$",
        xlim=(0, 1),
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
