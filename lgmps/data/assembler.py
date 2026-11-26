from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum
from itertools import accumulate
from typing import Literal

import torch

from .schema import SessionSequenceRecord

IGNORE_LABEL = -100


class TokenNamespace(IntEnum):
    QUERY = 0
    SEMANTIC = 1
    SPECIAL = 2


class TokenCategory(IntEnum):
    QUERY = 0
    ITEM = 1


class SpecialToken(IntEnum):
    QUERY_START = 0
    QUERY_END = 1
    ITEM_START = 2


@dataclass(slots=True)
class _Event:
    coordinate: int
    namespace: int
    token_id: int
    session_id: int
    item_id: int
    rope_position: int
    category: int
    is_positive_semantic: bool


@dataclass(slots=True)
class AssembledSequence:
    record_id: int
    user_id: int
    namespaces: torch.Tensor
    token_ids: torch.Tensor
    session_ids: torch.Tensor
    item_ids: torch.Tensor
    rope_positions: torch.Tensor
    categories: torch.Tensor
    train_labels: torch.Tensor
    validation_labels: torch.Tensor
    coordinates: torch.Tensor
    target_semantic_ids: list[tuple[int, ...]]

    @property
    def length(self) -> int:
        return int(self.token_ids.numel())


@dataclass(slots=True)
class PackedBatch:
    namespaces: torch.Tensor
    token_ids: torch.Tensor
    session_ids: torch.Tensor
    item_ids: torch.Tensor
    rope_positions: torch.Tensor
    categories: torch.Tensor
    train_labels: torch.Tensor
    validation_labels: torch.Tensor
    coordinates: torch.Tensor
    sequence_offsets: torch.Tensor
    session_offsets: torch.Tensor
    record_ids: torch.Tensor
    user_ids: torch.Tensor
    target_semantic_ids: list[list[tuple[int, ...]]]

    @property
    def token_count(self) -> int:
        return int(self.token_ids.numel())

    @property
    def batch_size(self) -> int:
        return int(self.record_ids.numel())

    def to(self, device: torch.device | str, non_blocking: bool = False) -> PackedBatch:
        tensor_names = (
            "namespaces",
            "token_ids",
            "session_ids",
            "item_ids",
            "rope_positions",
            "categories",
            "train_labels",
            "validation_labels",
            "coordinates",
            "sequence_offsets",
            "session_offsets",
            "record_ids",
            "user_ids",
        )
        values = {name: getattr(self, name).to(device, non_blocking=non_blocking) for name in tensor_names}
        values["target_semantic_ids"] = self.target_semantic_ids
        return PackedBatch(**values)

    def pin_memory(self) -> PackedBatch:
        tensor_names = (
            "namespaces",
            "token_ids",
            "session_ids",
            "item_ids",
            "rope_positions",
            "categories",
            "train_labels",
            "validation_labels",
            "coordinates",
            "sequence_offsets",
            "session_offsets",
            "record_ids",
            "user_ids",
        )
        values = {name: getattr(self, name).pin_memory() for name in tensor_names}
        values["target_semantic_ids"] = self.target_semantic_ids
        return PackedBatch(**values)


class SequenceAssembler:
    """Convert persisted scatter-coordinate records into compact chronological token streams."""

    def __init__(
        self,
        *,
        query_term_slots: int = 10,
        add_query_markers: bool = True,
        add_item_marker: bool = True,
        maximum_tokens: int = 4096,
        positive_event_mask: int = 0,
        target_policy: Literal["clicked", "exposed"] = "clicked",
    ) -> None:
        self.query_term_slots = query_term_slots
        self.add_query_markers = add_query_markers
        self.add_item_marker = add_item_marker
        self.maximum_tokens = maximum_tokens
        self.positive_event_mask = positive_event_mask
        self.target_policy = target_policy

    @property
    def query_span(self) -> int:
        return self.query_term_slots + (2 if self.add_query_markers else 0)

    def _is_positive(self, value: int) -> bool:
        if self.target_policy == "exposed":
            return True
        return bool(value) if self.positive_event_mask == 0 else bool(value & self.positive_event_mask)

    def _build_events(
        self,
        record: SessionSequenceRecord,
        mode: Literal["training", "test_prefix", "all"],
    ) -> tuple[list[_Event], list[tuple[int, ...]]]:
        session_count = record.session_count
        query_row_width = len(record.query_token_ids) // session_count
        query_scatter_width = len(record.query_scatter_positions) // session_count
        expected_scatter_width = query_row_width + (2 if self.add_query_markers else 0)
        if query_row_width != self.query_term_slots:
            raise ValueError(
                f"Record query width {query_row_width} disagrees with configured query_term_slots={self.query_term_slots}"
            )
        if query_scatter_width != expected_scatter_width:
            raise ValueError(
                f"Record query scatter width {query_scatter_width} disagrees with expected {expected_scatter_width}"
            )

        if mode == "training":
            full_item_sessions = session_count - 1
            query_sessions = full_item_sessions
        elif mode == "test_prefix":
            full_item_sessions = session_count - 1
            query_sessions = session_count
        else:
            full_item_sessions = session_count
            query_sessions = session_count

        semantic_depth = record.semantic_depth
        item_span = semantic_depth + (1 if self.add_item_marker else 0)
        session_stride = self.query_span + item_span
        events: list[_Event] = []

        for session_index in range(query_sessions):
            session_id = session_index + 1
            base = session_index * session_stride
            scatter_start = session_index * query_scatter_width
            scatter = record.query_scatter_positions[scatter_start : scatter_start + query_scatter_width]
            terms_start = session_index * query_row_width
            terms = record.query_token_ids[terms_start : terms_start + query_row_width]
            if self.add_query_markers:
                events.append(
                    _Event(
                        scatter[0],
                        TokenNamespace.SPECIAL,
                        SpecialToken.QUERY_START,
                        session_id,
                        0,
                        base,
                        TokenCategory.QUERY,
                        False,
                    )
                )
                term_positions = scatter[1:-1]
                end_position = scatter[-1]
                term_rope_start = 1
            else:
                term_positions = scatter
                end_position = None
                term_rope_start = 0
            for slot, (coordinate, token_id) in enumerate(zip(term_positions, terms, strict=True)):
                if token_id < 0:
                    continue
                events.append(
                    _Event(
                        coordinate,
                        TokenNamespace.QUERY,
                        token_id + 1,
                        session_id,
                        0,
                        base + term_rope_start + slot,
                        TokenCategory.QUERY,
                        False,
                    )
                )
            if end_position is not None:
                events.append(
                    _Event(
                        end_position,
                        TokenNamespace.SPECIAL,
                        SpecialToken.QUERY_END,
                        session_id,
                        0,
                        base + self.query_span - 1,
                        TokenCategory.QUERY,
                        False,
                    )
                )

        item_cursor = 0
        scatter_cursor = 0
        target_sids: list[tuple[int, ...]] = []
        for session_index, item_count in enumerate(record.items_per_session):
            session_id = session_index + 1
            include_items = session_index < full_item_sessions
            base = session_index * session_stride + self.query_span
            for local_item_index in range(item_count):
                semantic_row = record.semantic_ids[item_cursor]
                positive = self._is_positive(record.interaction_labels[item_cursor])
                item_id = local_item_index + 1
                item_scatter = record.item_scatter_positions[scatter_cursor : scatter_cursor + item_span]
                if (
                    session_index == session_count - 1
                    and positive
                    and all(value > 0 for value in semantic_row)
                ):
                    target_sids.append(tuple(int(value) for value in semantic_row))
                if include_items and any(value > 0 for value in semantic_row):
                    semantic_scatter = item_scatter
                    semantic_rope_start = 0
                    if self.add_item_marker:
                        events.append(
                            _Event(
                                item_scatter[0],
                                TokenNamespace.SPECIAL,
                                SpecialToken.ITEM_START,
                                session_id,
                                item_id,
                                base,
                                TokenCategory.ITEM,
                                False,
                            )
                        )
                        semantic_scatter = item_scatter[1:]
                        semantic_rope_start = 1
                    for level, (coordinate, token_id) in enumerate(
                        zip(semantic_scatter, semantic_row, strict=True)
                    ):
                        if token_id <= 0:
                            continue
                        events.append(
                            _Event(
                                coordinate,
                                TokenNamespace.SEMANTIC,
                                token_id,
                                session_id,
                                item_id,
                                base + semantic_rope_start + level,
                                TokenCategory.ITEM,
                                positive,
                            )
                        )
                item_cursor += 1
                scatter_cursor += item_span

        events.sort(key=lambda event: event.coordinate)
        coordinates = [event.coordinate for event in events]
        if len(coordinates) != len(set(coordinates)):
            raise ValueError(f"Record {record.record_id} has duplicate active scatter coordinates")
        return events, target_sids

    def assemble(
        self,
        record: SessionSequenceRecord,
        *,
        mode: Literal["training", "test_prefix", "all"] = "training",
    ) -> AssembledSequence:
        if mode in {"training", "test_prefix"} and record.session_count < 2:
            raise ValueError(
                f"Record {record.record_id} needs at least two sessions for chronological holdout"
            )
        events, targets = self._build_events(record, mode)
        if len(events) < 2:
            raise ValueError(f"Record {record.record_id} has fewer than two active tokens")
        if len(events) > self.maximum_tokens:
            raise ValueError(
                f"Record {record.record_id} has {len(events)} active tokens, above maximum_tokens={self.maximum_tokens}; "
                "re-run preprocessing with chronological truncation"
            )

        source = events if mode == "test_prefix" else events[:-1]
        next_events = None if mode == "test_prefix" else events[1:]
        train_cutoff = record.session_count - 2
        validation_session = record.session_count - 1
        train_labels = [IGNORE_LABEL] * len(source)
        validation_labels = [IGNORE_LABEL] * len(source)
        if next_events is not None:
            for index, next_event in enumerate(next_events):
                if not next_event.is_positive_semantic:
                    continue
                if next_event.session_id <= train_cutoff:
                    train_labels[index] = next_event.token_id
                elif next_event.session_id == validation_session:
                    validation_labels[index] = next_event.token_id

        def tensor(values: Iterable[int]) -> torch.Tensor:
            return torch.tensor(list(values), dtype=torch.long)

        return AssembledSequence(
            record_id=record.record_id,
            user_id=record.user_id,
            namespaces=tensor(event.namespace for event in source),
            token_ids=tensor(event.token_id for event in source),
            session_ids=tensor(event.session_id for event in source),
            item_ids=tensor(event.item_id for event in source),
            rope_positions=tensor(event.rope_position for event in source),
            categories=tensor(event.category for event in source),
            train_labels=tensor(train_labels),
            validation_labels=tensor(validation_labels),
            coordinates=tensor(event.coordinate for event in source),
            target_semantic_ids=targets,
        )


def _session_boundaries(sequences: list[AssembledSequence]) -> list[int]:
    boundaries = [0]
    cursor = 0
    for sequence in sequences:
        session_ids = sequence.session_ids.tolist()
        for index in range(1, len(session_ids)):
            if session_ids[index] != session_ids[index - 1]:
                boundaries.append(cursor + index)
        cursor += sequence.length
        boundaries.append(cursor)
    return sorted(set(boundaries))


def pack_sequences(sequences: list[AssembledSequence]) -> PackedBatch:
    if not sequences:
        raise ValueError("Cannot pack an empty sequence list")
    names = (
        "namespaces",
        "token_ids",
        "session_ids",
        "item_ids",
        "rope_positions",
        "categories",
        "train_labels",
        "validation_labels",
        "coordinates",
    )
    packed = {name: torch.cat([getattr(sequence, name) for sequence in sequences], dim=0) for name in names}
    sequence_offsets = torch.tensor(
        [0, *accumulate(sequence.length for sequence in sequences)], dtype=torch.int32
    )
    return PackedBatch(
        **packed,
        sequence_offsets=sequence_offsets,
        session_offsets=torch.tensor(_session_boundaries(sequences), dtype=torch.int32),
        record_ids=torch.tensor([sequence.record_id for sequence in sequences], dtype=torch.long),
        user_ids=torch.tensor([sequence.user_id for sequence in sequences], dtype=torch.long),
        target_semantic_ids=[sequence.target_semantic_ids for sequence in sequences],
    )
