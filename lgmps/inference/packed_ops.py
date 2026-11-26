from __future__ import annotations

from itertools import accumulate

import torch

from lgmps.data.assembler import IGNORE_LABEL, PackedBatch, TokenCategory, TokenNamespace


def append_token(
    batch: PackedBatch,
    *,
    namespace: int | TokenNamespace,
    token_id: int,
    item_id: int,
) -> PackedBatch:
    if batch.batch_size != 1:
        raise ValueError("append_token operates on one packed sequence")
    device = batch.token_ids.device

    def scalar(value: int, dtype: torch.dtype = torch.long) -> torch.Tensor:
        return torch.tensor([value], device=device, dtype=dtype)

    session_id = int(batch.session_ids[-1].item())
    category = int(TokenCategory.QUERY if item_id == 0 else TokenCategory.ITEM)
    fields = {
        "namespaces": torch.cat((batch.namespaces, scalar(int(namespace)))),
        "token_ids": torch.cat((batch.token_ids, scalar(token_id))),
        "session_ids": torch.cat((batch.session_ids, scalar(session_id))),
        "item_ids": torch.cat((batch.item_ids, scalar(item_id))),
        "rope_positions": torch.cat((batch.rope_positions, scalar(int(batch.rope_positions[-1]) + 1))),
        "categories": torch.cat((batch.categories, scalar(category))),
        "train_labels": torch.cat((batch.train_labels, scalar(IGNORE_LABEL))),
        "validation_labels": torch.cat((batch.validation_labels, scalar(IGNORE_LABEL))),
        "coordinates": torch.cat((batch.coordinates, scalar(int(batch.coordinates[-1]) + 1))),
    }
    sequence_offsets = batch.sequence_offsets.clone()
    session_offsets = batch.session_offsets.clone()
    sequence_offsets[-1] += 1
    session_offsets[-1] += 1
    return PackedBatch(
        **fields,
        sequence_offsets=sequence_offsets,
        session_offsets=session_offsets,
        record_ids=batch.record_ids,
        user_ids=batch.user_ids,
        target_semantic_ids=batch.target_semantic_ids,
    )


def combine_single_sequence_batches(batches: list[PackedBatch]) -> PackedBatch:
    if not batches or any(batch.batch_size != 1 for batch in batches):
        raise ValueError("Expected a non-empty list of single-sequence batches")
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
    packed = {name: torch.cat([getattr(batch, name) for batch in batches]) for name in names}
    lengths = [batch.token_count for batch in batches]
    sequence_offsets = torch.tensor(
        [0, *accumulate(lengths)],
        dtype=torch.int32,
        device=batches[0].token_ids.device,
    )
    session_values = [0]
    cursor = 0
    for batch in batches:
        session_values.extend(cursor + int(value) for value in batch.session_offsets[1:].detach().cpu())
        cursor += batch.token_count
    return PackedBatch(
        **packed,
        sequence_offsets=sequence_offsets,
        session_offsets=torch.tensor(session_values, dtype=torch.int32, device=batches[0].token_ids.device),
        record_ids=torch.cat([batch.record_ids for batch in batches]),
        user_ids=torch.cat([batch.user_ids for batch in batches]),
        target_semantic_ids=[batch.target_semantic_ids[0] for batch in batches],
    )
