from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def render_attention(
    artifact_file: str | Path,
    output_file: str | Path,
    *,
    layer: int = 0,
    session: int = -1,
    head: int = -1,
) -> Path:
    artifact = torch.load(artifact_file, map_location="cpu", weights_only=False)
    attention = artifact["attention"][layer]
    if attention is None:
        raise ValueError(f"Layer {layer} has no quadratic attention capture")
    matrix = attention[session].float()
    selected = matrix.mean(dim=0) if head < 0 else matrix[head]
    values = selected.numpy()
    positive = values[values > 0]
    floor = max(float(positive.min()) if positive.size else 1e-8, 1e-8)
    log_values = np.log10(np.maximum(values, floor))
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(log_values, origin="upper", aspect="auto", cmap="magma")
    axis.set_title(f"Attention, layer {layer}, session {session}, head {'mean' if head < 0 else head}")
    axis.set_xlabel("key token")
    axis.set_ylabel("query token")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("log10 attention probability")
    figure.tight_layout()
    destination = Path(output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=240, bbox_inches="tight")
    plt.close(figure)
    return destination
