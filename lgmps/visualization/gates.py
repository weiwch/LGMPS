from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same")


def _load_artifacts(source: str | Path) -> list[dict[str, Any]]:
    path = Path(source)
    files = sorted(path.glob("*.pt")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No gate artifacts under {path}")
    return [torch.load(file, map_location="cpu", weights_only=False) for file in files]


def render_gates(source: str | Path, output_dir: str | Path, *, smoothing: int = 21) -> dict[str, Any]:
    artifacts = _load_artifacts(source)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    by_layer: dict[int, dict[str, list[torch.Tensor]]] = {}
    for artifact in artifacts:
        for gate in artifact["gates"]:
            layer = int(gate["layer_idx"])
            values = by_layer.setdefault(
                layer, {"alpha": [], "beta": [], "alpha_logits": [], "beta_logits": []}
            )
            for name in values:
                if name in gate:
                    tensor = gate[name].float()
                    tensor = tensor.reshape(-1, tensor.shape[-1])
                    values[name].append(tensor)
    if not by_layer:
        raise ValueError("The artifacts contain no Gated DeltaNet captures")

    layers = sorted(by_layer)
    figure, axes = plt.subplots(len(layers), 2, figsize=(13, max(3.0, 2.7 * len(layers))), squeeze=False)
    alpha_head_means = []
    beta_head_means = []
    summary: dict[str, Any] = {"layers": {}}
    for row, layer in enumerate(layers):
        alpha = torch.cat(by_layer[layer]["alpha"], dim=0).numpy()
        beta = torch.cat(by_layer[layer]["beta"], dim=0).numpy()
        alpha_head_means.append(alpha.mean(axis=0))
        beta_head_means.append(beta.mean(axis=0))
        for column, (name, values) in enumerate((("retention alpha", alpha), ("update beta", beta))):
            series = values.mean(axis=-1)
            axes[row, column].plot(series, color="#3465a4", alpha=0.22, linewidth=0.55)
            axes[row, column].plot(_moving_average(series, smoothing), color="#cc4c02", linewidth=1.4)
            axes[row, column].set_ylim(-0.02, 1.02)
            axes[row, column].set_title(f"Layer {layer}: transformed {name}")
            axes[row, column].set_xlabel("packed token ordinal")
            axes[row, column].set_ylabel(name)
            axes[row, column].grid(alpha=0.16)
        summary["layers"][str(layer)] = {
            "tokens": int(alpha.shape[0]),
            "heads": int(alpha.shape[1]),
            "alpha_mean": float(alpha.mean()),
            "alpha_std": float(alpha.std()),
            "beta_mean": float(beta.mean()),
            "beta_std": float(beta.std()),
        }
    figure.tight_layout()
    series_path = output / "gdn_alpha_beta_series.png"
    figure.savefig(series_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    alpha_matrix = np.stack(alpha_head_means)
    beta_matrix = np.stack(beta_head_means)
    figure, axes = plt.subplots(1, 2, figsize=(11, max(3.5, 0.55 * len(layers) + 2)))
    for axis, matrix, title in (
        (axes[0], alpha_matrix, "Mean transformed alpha"),
        (axes[1], beta_matrix, "Mean transformed beta"),
    ):
        image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        axis.set_title(title)
        axis.set_xlabel("value head")
        axis.set_ylabel("layer")
        axis.set_yticks(range(len(layers)), labels=layers)
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    heatmap_path = output / "gdn_alpha_beta_head_heatmap.png"
    figure.savefig(heatmap_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    with (output / "gdn_gate_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {"series": str(series_path), "heatmap": str(heatmap_path), **summary}
