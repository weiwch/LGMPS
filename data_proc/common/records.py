from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "session-sequence.v1"
CANONICAL_FIELDS = (
    "schema_version",
    "record_id",
    "user_id",
    "dataset",
    "chunk_index",
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
    "item_identifiers",
)

QUERY_TOKEN_SLOTS = 10
QUERY_SPAN = QUERY_TOKEN_SLOTS + 2
SEMANTIC_DEPTH = 3
ITEM_SPAN = SEMANTIC_DEPTH + 1
VISIBILITY_ITEM_BITS = 32
VISIBILITY_ITEM_MASK = (1 << VISIBILITY_ITEM_BITS) - 1


@dataclass(frozen=True, slots=True)
class Document:
    item_identifier: str
    semantic_ids: tuple[int, int, int]
    interaction_label: int
    item_category: int

    def __post_init__(self) -> None:
        if not str(self.item_identifier).strip():
            raise ValueError("item_identifier must be non-empty")
        if len(self.semantic_ids) != SEMANTIC_DEPTH:
            raise ValueError(f"semantic_ids must contain {SEMANTIC_DEPTH} levels")


@dataclass(frozen=True, slots=True)
class Session:
    query_token_ids: tuple[int, ...]
    session_attribute: int
    documents: tuple[Document, ...]

    def __post_init__(self) -> None:
        if len(self.query_token_ids) > QUERY_TOKEN_SLOTS:
            raise ValueError(f"query_token_ids exceeds {QUERY_TOKEN_SLOTS} tokens")
        if not self.documents:
            raise ValueError("a session must contain at least one document")


@dataclass(slots=True)
class WriteSummary:
    users: int = 0
    source_sessions: int = 0
    split_sessions: int = 0
    written_records: int = 0
    written_sessions: int = 0
    written_items: int = 0
    filtered_short_chunks: int = 0


def _visibility_code(session_ordinal: int, item_ordinal: int) -> int:
    """Pack a one-based session and item ordinal without collisions."""
    if session_ordinal <= 0:
        raise ValueError("session ordinal must be positive")
    if item_ordinal < 0 or item_ordinal > VISIBILITY_ITEM_MASK:
        raise ValueError("item ordinal is outside the visibility encoding range")
    return (session_ordinal << VISIBILITY_ITEM_BITS) | item_ordinal


def _split_oversized_session(session: Session, maximum_tokens: int) -> list[Session]:
    maximum_items = (maximum_tokens - QUERY_SPAN) // ITEM_SPAN
    if maximum_items < 1:
        raise ValueError(
            f"maximum_tokens={maximum_tokens} cannot hold one {QUERY_SPAN}-token request and one item"
        )
    if len(session.documents) <= maximum_items:
        return [session]
    return [
        Session(
            query_token_ids=session.query_token_ids,
            session_attribute=session.session_attribute,
            documents=session.documents[start : start + maximum_items],
        )
        for start in range(0, len(session.documents), maximum_items)
    ]


def _chunk_sessions(sessions: Sequence[Session], maximum_tokens: int) -> tuple[list[list[Session]], int]:
    expanded: list[Session] = []
    split_count = 0
    for session in sessions:
        parts = _split_oversized_session(session, maximum_tokens)
        expanded.extend(parts)
        split_count += len(parts) - 1

    chunks: list[list[Session]] = []
    current: list[Session] = []
    current_tokens = 0
    for session in expanded:
        session_tokens = QUERY_SPAN + ITEM_SPAN * len(session.documents)
        if current and current_tokens + session_tokens > maximum_tokens:
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append(session)
        current_tokens += session_tokens
    if current:
        chunks.append(current)
    return chunks, split_count


def _materialize_record(
    *,
    record_id: int,
    user_id: int,
    dataset: str,
    chunk_index: int,
    sessions: Sequence[Session],
) -> dict[str, Any]:
    semantic_ids: list[list[int]] = []
    items_per_session: list[int] = []
    query_token_ids: list[int] = []
    query_lengths: list[int] = []
    interaction_labels: list[int] = []
    session_attributes: list[int] = []
    item_categories: list[int] = []
    visibility_codes: list[int] = []
    query_scatter_positions: list[int] = []
    item_scatter_positions: list[int] = []
    semantic_target_positions: list[int] = []
    item_identifiers: list[str] = []
    cursor = 0

    for session_ordinal, session in enumerate(sessions, start=1):
        terms = [int(value) for value in session.query_token_ids[:QUERY_TOKEN_SLOTS]]
        query_lengths.append(len(terms))
        query_token_ids.extend([*terms, *([-1] * (QUERY_TOKEN_SLOTS - len(terms)))])
        session_attributes.append(int(session.session_attribute))
        items_per_session.append(len(session.documents))

        query_code = _visibility_code(session_ordinal, 0)
        query_scatter_positions.extend(range(cursor, cursor + QUERY_SPAN))
        visibility_codes.extend([query_code] * QUERY_SPAN)
        cursor += QUERY_SPAN

        for item_ordinal, document in enumerate(session.documents, start=1):
            semantic_ids.append([int(value) for value in document.semantic_ids])
            interaction_labels.append(int(document.interaction_label))
            item_categories.append(int(document.item_category))
            item_identifiers.append(str(document.item_identifier))

            item_code = _visibility_code(session_ordinal, item_ordinal)
            item_scatter_positions.extend(range(cursor, cursor + ITEM_SPAN))
            # These are the coordinates of the SID tokens themselves, after the item marker.
            semantic_target_positions.extend(range(cursor + 1, cursor + ITEM_SPAN))
            visibility_codes.extend([item_code] * ITEM_SPAN)
            cursor += ITEM_SPAN

    item_count = len(semantic_ids)
    record = {
        "schema_version": SCHEMA_VERSION,
        "record_id": int(record_id),
        "user_id": int(user_id),
        "dataset": str(dataset),
        "chunk_index": int(chunk_index),
        "item_token_count": item_count * ITEM_SPAN,
        "query_slot_count": len(sessions) * QUERY_SPAN,
        "semantic_ids": semantic_ids,
        "items_per_session": items_per_session,
        "query_token_ids": query_token_ids,
        "query_lengths": query_lengths,
        "interaction_labels": interaction_labels,
        "session_attributes": session_attributes,
        "item_categories": item_categories,
        "visibility_codes": visibility_codes,
        "query_scatter_positions": query_scatter_positions,
        "item_scatter_positions": item_scatter_positions,
        "semantic_target_positions": semantic_target_positions,
        "item_identifiers": item_identifiers,
    }
    validate_record(record)
    return record


def validate_record(record: dict[str, Any]) -> None:
    if tuple(record) != CANONICAL_FIELDS:
        missing = sorted(set(CANONICAL_FIELDS) - set(record))
        extra = sorted(set(record) - set(CANONICAL_FIELDS))
        raise ValueError(f"canonical field mismatch; missing={missing}, extra={extra}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version={record['schema_version']!r}")
    if any(int(record[name]) < 0 for name in ("record_id", "user_id", "chunk_index")):
        raise ValueError("record_id, user_id, and chunk_index must be non-negative")
    if not record["dataset"]:
        raise ValueError("dataset must be non-empty")

    session_count = len(record["items_per_session"])
    item_count = len(record["semantic_ids"])
    if session_count < 1 or any(int(count) < 1 for count in record["items_per_session"]):
        raise ValueError("every record and session must be non-empty")
    if sum(record["items_per_session"]) != item_count:
        raise ValueError("items_per_session does not sum to the item count")
    if any(len(row) != SEMANTIC_DEPTH for row in record["semantic_ids"]):
        raise ValueError("semantic_ids must be a three-column matrix")
    for row in record["semantic_ids"]:
        if any(int(value) < 0 for value in row):
            raise ValueError("semantic IDs must be non-negative")

    if len(record["query_token_ids"]) != session_count * QUERY_TOKEN_SLOTS:
        raise ValueError("query_token_ids does not contain one fixed-width row per session")
    if len(record["query_lengths"]) != session_count:
        raise ValueError("query_lengths must contain one value per session")
    if any(not 0 <= int(length) <= QUERY_TOKEN_SLOTS for length in record["query_lengths"]):
        raise ValueError("query length is outside the fixed query width")
    if len(record["session_attributes"]) != session_count:
        raise ValueError("session_attributes must contain one value per session")

    for name in ("interaction_labels", "item_categories", "item_identifiers"):
        if len(record[name]) != item_count:
            raise ValueError(f"{name} must contain one value per item")
    if any(not str(identifier).strip() for identifier in record["item_identifiers"]):
        raise ValueError("item_identifiers cannot contain null or empty values")

    item_token_count = item_count * ITEM_SPAN
    query_slot_count = session_count * QUERY_SPAN
    total_coordinates = item_token_count + query_slot_count
    if record["item_token_count"] != item_token_count:
        raise ValueError("item_token_count is inconsistent")
    if record["query_slot_count"] != query_slot_count:
        raise ValueError("query_slot_count is inconsistent")
    if len(record["visibility_codes"]) != total_coordinates:
        raise ValueError("visibility_codes must cover every dense coordinate")
    if len(record["query_scatter_positions"]) != query_slot_count:
        raise ValueError("query_scatter_positions has the wrong length")
    if len(record["item_scatter_positions"]) != item_token_count:
        raise ValueError("item_scatter_positions has the wrong length")
    if len(record["semantic_target_positions"]) != item_count * SEMANTIC_DEPTH:
        raise ValueError("semantic_target_positions has the wrong length")

    expected_targets: list[int] = []
    for start in range(0, item_token_count, ITEM_SPAN):
        item_positions = record["item_scatter_positions"][start : start + ITEM_SPAN]
        expected_targets.extend(item_positions[1:])
    if record["semantic_target_positions"] != expected_targets:
        raise ValueError("semantic_target_positions must point to the three SID coordinates")

    all_scatter = [*record["query_scatter_positions"], *record["item_scatter_positions"]]
    if sorted(all_scatter) != list(range(total_coordinates)):
        raise ValueError("scatter positions must partition the dense coordinate range")


class CanonicalJsonlWriter:
    """Atomically write validated, chronologically chunked canonical records."""

    def __init__(
        self,
        output_file: str | Path,
        *,
        maximum_tokens: int = 4096,
        minimum_sessions: int = 2,
        overwrite: bool = False,
    ) -> None:
        if maximum_tokens < QUERY_SPAN + ITEM_SPAN:
            raise ValueError("maximum_tokens is too small")
        if minimum_sessions < 1:
            raise ValueError("minimum_sessions must be at least one")
        self.output_path = Path(output_file).expanduser().resolve()
        if self.output_path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite {self.output_path}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary_path = self.output_path.with_name(f".{self.output_path.name}.tmp.{os.getpid()}")
        self.maximum_tokens = int(maximum_tokens)
        self.minimum_sessions = int(minimum_sessions)
        self.summary = WriteSummary()
        self._handle = None
        self._next_record_id = 0

    def __enter__(self) -> CanonicalJsonlWriter:  # noqa: PYI034 - Python 3.10 support
        self._handle = self.temporary_path.open("w", encoding="utf-8")
        return self

    def write_user(self, *, user_id: int, dataset: str, sessions: Sequence[Session]) -> None:
        if self._handle is None:
            raise RuntimeError("writer must be used as a context manager")
        if int(user_id) < 0:
            raise ValueError("user_id must be non-negative")
        self.summary.users += 1
        self.summary.source_sessions += len(sessions)
        chunks, split_count = _chunk_sessions(sessions, self.maximum_tokens)
        self.summary.split_sessions += split_count
        for chunk_index, chunk in enumerate(chunks):
            if len(chunk) < self.minimum_sessions:
                self.summary.filtered_short_chunks += 1
                continue
            record = _materialize_record(
                record_id=self._next_record_id,
                user_id=int(user_id),
                dataset=dataset,
                chunk_index=chunk_index,
                sessions=chunk,
            )
            self._handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            self.summary.written_records += 1
            self.summary.written_sessions += len(chunk)
            self.summary.written_items += len(record["semantic_ids"])
            self._next_record_id += 1

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        if exc_type is None:
            os.replace(self.temporary_path, self.output_path)
        elif self.temporary_path.exists():
            self.temporary_path.unlink()

    def summary_dict(self) -> dict[str, int]:
        return {key: int(value) for key, value in asdict(self.summary).items()}
