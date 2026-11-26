from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

from .io import reject_lfs_pointer

SID_FIELDS = ("item_identifier", "semantic_id_1", "semantic_id_2", "semantic_id_3")


def read_semantic_id_map(
    path: str | Path,
    *,
    num_centroids: int = 256,
) -> dict[str, tuple[int, int, int]]:
    resolved = reject_lfs_pointer(path)
    semantic_ids: dict[str, tuple[int, int, int]] = {}
    with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SID_FIELDS:
            raise ValueError(f"{resolved} must have exactly this header: {','.join(SID_FIELDS)}")
        for line_number, row in enumerate(reader, start=2):
            identifier = str(row["item_identifier"]).strip()
            if not identifier:
                raise ValueError(f"{resolved}:{line_number} has an empty item_identifier")
            if identifier in semantic_ids:
                raise ValueError(f"{resolved}:{line_number} duplicates item_identifier={identifier!r}")
            try:
                raw = tuple(int(row[f"semantic_id_{level}"]) for level in range(1, 4))
            except (TypeError, ValueError) as error:
                raise ValueError(f"{resolved}:{line_number} has a non-integer semantic ID") from error
            if any(value < 0 or value >= num_centroids for value in raw):
                raise ValueError(
                    f"{resolved}:{line_number} semantic IDs must be zero-based values below {num_centroids}"
                )
            semantic_ids[identifier] = (
                raw[0] + 1,
                raw[1] + num_centroids + 1,
                raw[2] + num_centroids * 2 + 1,
            )
    if not semantic_ids:
        raise ValueError(f"{resolved} contains no semantic IDs")
    return semantic_ids


def semantic_ids_for(
    item_identifier: str,
    semantic_id_map: Mapping[str, tuple[int, int, int]],
) -> tuple[int, int, int]:
    return semantic_id_map.get(str(item_identifier), (0, 0, 0))


def write_semantic_ids(
    path: str | Path,
    identifiers: Sequence[str],
    codes: Sequence[Sequence[int]],
) -> None:
    if len(identifiers) != len(codes):
        raise ValueError("identifiers and codes have different lengths")
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SID_FIELDS)
        writer.writeheader()
        seen: set[str] = set()
        for identifier, row in zip(identifiers, codes, strict=True):
            identifier = str(identifier).strip()
            if not identifier or identifier in seen:
                raise ValueError(f"invalid or duplicate item_identifier={identifier!r}")
            if len(row) != 3:
                raise ValueError("RQ codes must contain exactly three levels")
            seen.add(identifier)
            writer.writerow(
                {
                    "item_identifier": identifier,
                    "semantic_id_1": int(row[0]),
                    "semantic_id_2": int(row[1]),
                    "semantic_id_3": int(row[2]),
                }
            )
