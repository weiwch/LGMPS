from __future__ import annotations

import pytest

from lgmps.data.schema import SCHEMA_VERSION, SessionSequenceRecord


@pytest.fixture
def toy_record() -> SessionSequenceRecord:
    sessions = 3
    items_per_session = [2, 2, 2]
    query_tokens: list[int] = []
    query_lengths: list[int] = []
    semantic_ids: list[list[int]] = []
    interactions: list[int] = []
    item_categories: list[int] = []
    visibility: list[int] = []
    query_scatter: list[int] = []
    item_scatter: list[int] = []
    target_positions: list[int] = []
    cursor = 0
    item_ordinal = 0
    for session in range(sessions):
        terms = [10 + session, 20 + session, *([-1] * 8)]
        query_tokens.extend(terms)
        query_lengths.append(2)
        query_scatter.extend(range(cursor, cursor + 12))
        visibility.extend([(session + 1) * 1000] * 12)
        cursor += 12
        for local_item in range(2):
            first = 1 + item_ordinal
            semantic_ids.append([first, 256 + first, 512 + first])
            interactions.append(1 if local_item == 0 else 0)
            item_categories.append(2 if local_item == 0 else 1)
            item_scatter.extend(range(cursor, cursor + 4))
            target_positions.extend(range(cursor, cursor + 3))
            visibility.extend([(session + 1) * 1000 + local_item + 1] * 4)
            cursor += 4
            item_ordinal += 1
    return SessionSequenceRecord(
        schema_version=SCHEMA_VERSION,
        record_id=0,
        user_id=7,
        dataset="KuaiSAR_v2",
        chunk_index=0,
        item_token_count=24,
        query_slot_count=36,
        semantic_ids=semantic_ids,
        items_per_session=items_per_session,
        query_token_ids=query_tokens,
        query_lengths=query_lengths,
        interaction_labels=interactions,
        session_attributes=[1, 2, 1],
        item_categories=item_categories,
        visibility_codes=visibility,
        query_scatter_positions=query_scatter,
        item_scatter_positions=item_scatter,
        semantic_target_positions=target_positions,
    )
