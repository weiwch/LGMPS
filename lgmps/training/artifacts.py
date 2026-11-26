from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


class ArtifactSink:
    def __init__(self, output_dir: str | Path, *, enabled: bool = True) -> None:
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        if enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "gates").mkdir(exist_ok=True)

    def write_jsonl(self, name: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        with (self.output_dir / name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def write_json(self, name: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        destination = self.output_dir / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(destination)

    def write_gate_capture(
        self,
        step: int,
        records: list[dict[str, Any]],
        metadata: dict[str, torch.Tensor],
    ) -> None:
        if not self.enabled:
            return
        payload = {
            "step": step,
            "gates": [
                {
                    key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                    for key, value in record.items()
                }
                for record in records
            ],
            "metadata": {name: value.detach().cpu() for name, value in metadata.items()},
        }
        torch.save(payload, self.output_dir / "gates" / f"step_{step:08d}.pt")
