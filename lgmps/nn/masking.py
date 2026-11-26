from __future__ import annotations

from itertools import pairwise
from typing import Literal

import torch


def build_session_visibility(item_ids: torch.Tensor) -> torch.Tensor:
    """Paper mask for one session: causal and query-side-or-same-item visibility."""
    if item_ids.ndim != 1:
        raise ValueError("item_ids must be one-dimensional")
    length = item_ids.numel()
    positions = torch.arange(length, device=item_ids.device)
    causal = positions[:, None] >= positions[None, :]
    query_side_key = item_ids[None, :] == 0
    same_item = item_ids[:, None] == item_ids[None, :]
    return causal & (query_side_key | same_item)


def build_session_causal_visibility(
    item_ids: torch.Tensor,
    session_ids: torch.Tensor,
) -> torch.Tensor:
    """Session-causal visibility for one user sequence.

    A token can read every causal token in an earlier session.  Inside its own
    session it can read request tokens and tokens belonging to the same item,
    but not another item from that request.
    """
    if item_ids.ndim != 1 or session_ids.shape != item_ids.shape:
        raise ValueError("item_ids and session_ids must be matching one-dimensional tensors")
    length = item_ids.numel()
    positions = torch.arange(length, device=item_ids.device)
    causal = positions[:, None] >= positions[None, :]
    previous_session = session_ids[:, None] > session_ids[None, :]
    same_session = session_ids[:, None] == session_ids[None, :]
    query_side_key = item_ids[None, :] == 0
    same_item = item_ids[:, None] == item_ids[None, :]
    return causal & (previous_session | (same_session & (query_side_key | same_item)))


def segmented_attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    offsets: torch.Tensor,
    *,
    item_ids: torch.Tensor | None = None,
    session_ids: torch.Tensor | None = None,
    pattern: Literal["session", "session_causal", "causal"] = "session",
    scale: float | None = None,
    return_weights: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    """Differentiable PyTorch oracle for packed variable-length attention."""
    if query.ndim != 3 or key.shape != query.shape or value.shape != query.shape:
        raise ValueError("query, key, and value must share shape [tokens, heads, head_width]")
    if offsets.ndim != 1 or offsets.numel() < 2:
        raise ValueError("offsets must contain at least [0, end]")
    if int(offsets[0]) != 0 or int(offsets[-1]) != query.shape[0]:
        raise ValueError("offsets must exactly cover the packed token axis")
    if pattern == "session" and (item_ids is None or item_ids.shape != (query.shape[0],)):
        raise ValueError("session attention requires one item id per token")
    if pattern == "session_causal" and (
        item_ids is None
        or item_ids.shape != (query.shape[0],)
        or session_ids is None
        or session_ids.shape != (query.shape[0],)
    ):
        raise ValueError("session-causal attention requires one item and session id per token")
    resolved_scale = scale if scale is not None else query.shape[-1] ** -0.5
    outputs: list[torch.Tensor] = []
    captured: list[torch.Tensor] = []
    offset_values = offsets.detach().to(device="cpu", dtype=torch.int64).tolist()
    for start, end in pairwise(offset_values):
        if end <= start:
            continue
        q = query[start:end].transpose(0, 1)
        k = key[start:end].transpose(0, 1)
        v = value[start:end].transpose(0, 1)
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * resolved_scale
        if pattern == "session":
            visible = build_session_visibility(item_ids[start:end])
        elif pattern == "session_causal":
            visible = build_session_causal_visibility(
                item_ids[start:end],
                session_ids[start:end],
            )
        else:
            visible = torch.ones((end - start, end - start), dtype=torch.bool, device=query.device).tril_()
        probabilities = torch.softmax(scores.masked_fill(~visible.unsqueeze(0), -torch.inf), dim=-1)
        output = torch.matmul(probabilities.to(v.dtype), v).transpose(0, 1)
        outputs.append(output)
        if return_weights:
            captured.append(probabilities.detach())
    if not outputs:
        result = query.new_empty(query.shape)
    else:
        result = torch.cat(outputs, dim=0)
    return (result, captured) if return_weights else result
