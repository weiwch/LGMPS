from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def render_moe(log_file: str | Path, output_dir: str | Path) -> list[Path]:
    records = []
    with Path(log_file).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise ValueError("MoE log is empty")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    layer_count = len(records[0]["layers"])
    outputs: list[Path] = []
    for layer in range(layer_count):
        rows = [record["layers"][layer] for record in records if record["layers"][layer] is not None]
        if not rows:
            continue
        counts = np.asarray([row["counts"] for row in rows], dtype=np.float64)
        category = np.asarray([row["category_counts"] for row in rows], dtype=np.float64)
        steps = [record["step"] for record in records if record["layers"][layer] is not None]
        figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        image = axes[0].imshow(counts.T, aspect="auto", origin="lower", cmap="Blues")
        axes[0].set_ylabel("expert")
        axes[0].set_title(f"Layer {layer} executed route counts")
        figure.colorbar(image, ax=axes[0], fraction=0.025)
        query_total = category[:, 0, :].sum(axis=1)
        item_total = category[:, 1, :].sum(axis=1)
        axes[1].plot(steps, query_total, label="query routes")
        axes[1].plot(steps, item_total, label="item routes")
        axes[1].set_xlabel("optimizer step")
        axes[1].set_ylabel("executed routes")
        axes[1].legend()
        axes[1].grid(alpha=0.2)
        figure.tight_layout()
        output = destination / f"moe_layer_{layer:02d}.png"
        figure.savefig(output, dpi=220, bbox_inches="tight")
        plt.close(figure)
        outputs.append(output)
    return outputs
