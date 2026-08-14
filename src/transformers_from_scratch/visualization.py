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
