"""Attention-map rendering kept separate from the training code."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def save_attention_maps(
    attention_maps: list[torch.Tensor],
    tokens: list[str],
    output_dir: str | Path,
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    token_count = len(tokens)

    for layer_index, layer_map in enumerate(attention_maps, start=1):
        heads = layer_map[0, :, :token_count, :token_count].detach().cpu()
        for head_index, head_map in enumerate(heads, start=1):
            figure, axis = plt.subplots(figsize=(7, 6))
            image = axis.imshow(head_map, cmap="magma", vmin=0.0)
            axis.set_xticks(range(token_count), tokens, rotation=90)
            axis.set_yticks(range(token_count), tokens)
            axis.set_xlabel("Key token")
            axis.set_ylabel("Query token")
            axis.set_title(f"Layer {layer_index}, head {head_index}")
            figure.colorbar(image, ax=axis, label="Attention probability")
            figure.tight_layout()
            destination = output_path / f"layer_{layer_index}_head_{head_index}.png"
            figure.savefig(destination, dpi=160)
            plt.close(figure)
            written.append(destination)
    return written


def save_attention_overview(
    attention_maps: list[torch.Tensor],
    tokens: list[str],
    output: str | Path,
) -> Path:
    """Save every layer and head in one publication-ready attention figure."""
    if not attention_maps:
        raise ValueError("attention_maps cannot be empty")
    token_count = len(tokens)
    if token_count == 0:
        raise ValueError("tokens cannot be empty")

    layers = len(attention_maps)
    heads = attention_maps[0].shape[1]
    values = [
        layer[0, :, :token_count, :token_count].detach().float().cpu()
        for layer in attention_maps
    ]
    shared_maximum = max(float(layer.max()) for layer in values)
    figure, axes = plt.subplots(
        heads,
        layers,
        figsize=(3.5 * layers, 3.8 * heads),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    for layer_index, layer in enumerate(values):
        for head_index in range(heads):
            axis = axes[head_index, layer_index]
            image = axis.imshow(
                layer[head_index].numpy(),
                cmap="magma",
                vmin=0.0,
                vmax=shared_maximum,
                interpolation="nearest",
            )
            axis.set_title(f"Layer {layer_index + 1} · Head {head_index + 1}", weight="semibold")
            axis.set_xticks(range(token_count), tokens, rotation=55, ha="right")
            axis.set_yticks(range(token_count), tokens)
            axis.tick_params(axis="both", labelsize=8)
            if layer_index == 0:
                axis.set_ylabel("Query token")
            if head_index == heads - 1:
                axis.set_xlabel("Key token")

    assert image is not None
    figure.colorbar(image, ax=axes, shrink=0.55, pad=0.025, label="Attention probability")
    figure.suptitle(
        "Learned attention across the absolute-position encoder",
        fontsize=15,
        weight="bold",
    )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return destination
