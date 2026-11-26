from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

SCHEMA_VERSION = "session-sequence.v1"

REQUIRED_FIELDS = {
    "record_id",
    "user_id",
    "item_token_count",
    "query_slot_count",
    "semantic_ids",
    "items_per_session",
    "query_token_ids",
    "query_lengths",
    "interaction_labels",
    "session_attributes",
    "item_categories",
    "visibility_codes",
    "query_scatter_positions",
    "item_scatter_positions",
    "semantic_target_positions",
}


@dataclass(slots=True)
class SessionSequenceRecord:
    record_id: int
    user_id: int
    item_token_count: int
    query_slot_count: int
    semantic_ids: list[list[int]]
    items_per_session: list[int]
    query_token_ids: list[int]
    query_lengths: list[int]
    interaction_labels: list[int]
    session_attributes: list[int]
    item_categories: list[int]
    visibility_codes: list[int]
    query_scatter_positions: list[int]
    item_scatter_positions: list[int]
    semantic_target_positions: list[int]
    dataset: str = "KuaiSAR_v2"
    chunk_index: int = 0
    item_identifiers: list[str] | None = None
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        record_id: int | None = None,
        dataset: str | None = None,
        validate: bool = True,
    ) -> SessionSequenceRecord:
        is_canonical = "schema_version" in payload
        normalized = dict(payload)
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(normalized) - known)
        if unknown:
            raise ValueError(f"Unknown sequence fields: {', '.join(unknown)}")
        if is_canonical:
            missing_canonical = sorted(known - normalized.keys())
            if missing_canonical:
                raise ValueError(f"Missing sequence fields: {', '.join(missing_canonical)}")
        if "record_id" not in normalized and record_id is not None:
            normalized["record_id"] = int(record_id)
        if "dataset" not in normalized and dataset:
            normalized["dataset"] = dataset
        missing = sorted(REQUIRED_FIELDS - normalized.keys())
        if missing:
            raise ValueError(f"Missing sequence fields: {', '.join(missing)}")
        record = cls(**{name: normalized[name] for name in cls.__dataclass_fields__ if name in normalized})
        if validate:
            record.validate()
        return record

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "user_id": self.user_id,
            "dataset": self.dataset,
            "chunk_index": self.chunk_index,
            "item_token_count": self.item_token_count,
            "query_slot_count": self.query_slot_count,
            "semantic_ids": self.semantic_ids,
            "items_per_session": self.items_per_session,
            "query_token_ids": self.query_token_ids,
            "query_lengths": self.query_lengths,
            "interaction_labels": self.interaction_labels,
            "session_attributes": self.session_attributes,
            "item_categories": self.item_categories,
            "visibility_codes": self.visibility_codes,
            "query_scatter_positions": self.query_scatter_positions,
            "item_scatter_positions": self.item_scatter_positions,
            "semantic_target_positions": self.semantic_target_positions,
            "item_identifiers": self.item_identifiers,
        }

    @property
    def session_count(self) -> int:
        return len(self.items_per_session)

    @property
    def item_count(self) -> int:
        return len(self.semantic_ids)

    @property
    def semantic_depth(self) -> int:
        return len(self.semantic_ids[0]) if self.semantic_ids else 0

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version={self.schema_version!r}")
        if not self.dataset:
            raise ValueError("dataset must be a non-empty name")
        if self.record_id < 0 or self.user_id < 0 or self.chunk_index < 0:
            raise ValueError("record_id, user_id, and chunk_index must be non-negative")
        if self.session_count < 1:
            raise ValueError("A record needs at least one session")
        if sum(self.items_per_session) != self.item_count:
            raise ValueError("items_per_session does not sum to semantic_ids length")
        if self.semantic_depth <= 0 or any(len(row) != self.semantic_depth for row in self.semantic_ids):
            raise ValueError("semantic_ids must be a rectangular non-empty matrix")
        item_fields = (self.interaction_labels, self.item_categories)
        if any(len(values) != self.item_count for values in item_fields):
            raise ValueError("Item-level side fields must match semantic_ids length")
        if self.item_identifiers is not None and len(self.item_identifiers) != self.item_count:
            raise ValueError("item_identifiers must match semantic_ids length when supplied")
        if len(self.query_lengths) != self.session_count:
            raise ValueError("query_lengths must have one entry per session")
        if len(self.session_attributes) != self.session_count:
            raise ValueError("session_attributes must have one entry per session")
        if len(self.query_token_ids) % self.session_count:
            raise ValueError("query_token_ids must contain a fixed-width row per session")
        if len(self.query_scatter_positions) % self.session_count:
            raise ValueError("query_scatter_positions must contain a fixed-width row per session")
        item_span = self.semantic_depth + 1
        if len(self.item_scatter_positions) != self.item_count * item_span:
            raise ValueError("item_scatter_positions must contain marker + semantic levels per item")
        if len(self.semantic_target_positions) != self.item_count * self.semantic_depth:
            raise ValueError("semantic_target_positions must contain one coordinate per semantic token")
        total_coordinates = self.item_token_count + self.query_slot_count
        if len(self.visibility_codes) < total_coordinates:
            raise ValueError("visibility_codes is shorter than the declared flattened sequence")
        for name, positions in (
            ("query_scatter_positions", self.query_scatter_positions),
            ("item_scatter_positions", self.item_scatter_positions),
            ("semantic_target_positions", self.semantic_target_positions),
        ):
            if positions and (min(positions) < 0 or max(positions) >= len(self.visibility_codes)):
                raise ValueError(f"{name} contains an out-of-range coordinate")
