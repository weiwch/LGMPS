from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .indexed_jsonl import build_offset_index
from .schema import SCHEMA_VERSION, SessionSequenceRecord

try:
    import orjson
except ImportError:  # pragma: no cover
    orjson = None


@dataclass(slots=True)
class ConversionSummary:
    source_records: int = 0
    written_records: int = 0
    filtered_short_history: int = 0
    invalid_records: int = 0
    users: int = 0
    sessions: int = 0
    items: int = 0
    positive_items: int = 0
    missing_sid_items: int = 0


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def _loads(raw: bytes) -> dict[str, Any]:
    return orjson.loads(raw) if orjson is not None else json.loads(raw)


def _dumps(payload: dict[str, Any]) -> bytes:
    if orjson is not None:
        return orjson.dumps(payload, option=orjson.OPT_APPEND_NEWLINE)
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")


def convert_legacy_jsonl(
    source_file: str | Path,
    output_file: str | Path,
    *,
    index_file: str | Path | None = None,
    minimum_sessions: int = 2,
    overwrite: bool = False,
    skip_invalid: bool = False,
) -> ConversionSummary:
    """Convert unversioned KuaiSAR records that use the public field names."""
    source_path = Path(source_file).expanduser().resolve()
    output_path = Path(output_file).expanduser().resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    summary = ConversionSummary()
    per_user_chunks: Counter[int] = Counter()
    unique_users: set[int] = set()
    try:
        with source_path.open("rb") as source, temporary.open("wb") as target:
            for line_number, raw in enumerate(source, start=1):
                if not raw.strip():
                    continue
                summary.source_records += 1
                payload = _loads(raw)
                user_id = int(payload.get("user_id", -1))
                chunk_index = per_user_chunks[user_id]
                per_user_chunks[user_id] += 1
                payload["dataset"] = "KuaiSAR_v2"
                payload["chunk_index"] = chunk_index
                try:
                    record = SessionSequenceRecord.from_mapping(
                        payload,
                        record_id=summary.written_records,
                        validate=True,
                    )
                except (TypeError, ValueError, KeyError):
                    summary.invalid_records += 1
                    if skip_invalid:
                        continue
                    raise ValueError(f"Invalid source record at line {line_number}") from None
                if record.session_count < minimum_sessions:
                    summary.filtered_short_history += 1
                    continue
                record.record_id = summary.written_records
                target.write(_dumps(record.to_mapping()))
                summary.written_records += 1
                unique_users.add(record.user_id)
                summary.sessions += record.session_count
                summary.items += record.item_count
                summary.positive_items += sum(bool(value) for value in record.interaction_labels)
                summary.missing_sid_items += sum(not any(row) for row in record.semantic_ids)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    summary.users = len(unique_users)
    resolved_index = build_offset_index(output_path, index_file)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "KuaiSAR_v2",
        "source_file": str(source_path),
        "source_sha256": sha256_file(source_path),
        "output_file": str(output_path),
        "output_sha256": sha256_file(output_path),
        "index_file": str(resolved_index),
        "minimum_sessions": minimum_sessions,
        "semantic_id_encoding": {
            "missing": 0,
            "level_1": [1, 256],
            "level_2": [257, 512],
            "level_3": [513, 768],
        },
        "chronological_holdout": "last session is test; penultimate session is validation",
        "summary": asdict(summary),
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary
