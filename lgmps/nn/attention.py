from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from .masking import segmented_attention_reference
from .rope import apply_rotary
from .triton_session_attention import session_flash_attention

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except ImportError:  # pragma: no cover - depends on the installed PyTorch version
    create_block_mask = None
    flex_attention = None


_compiled_flex_attention = torch.compile(flex_attention, dynamic=True) if flex_attention is not None else None


@dataclass(slots=True)
class SessionAttentionCache:
    """Projected states used by one-token structured-attention beam decoding."""

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    item_ids: torch.Tensor
    session_ids: torch.Tensor

    def index_select(self, indices: torch.Tensor) -> SessionAttentionCache:
        return SessionAttentionCache(
            query=self.query.index_select(0, indices),
            key=self.key.index_select(0, indices),
            value=self.value.index_select(0, indices),
            item_ids=self.item_ids.index_select(0, indices),
            session_ids=self.session_ids.index_select(0, indices),
        )


def build_session_local_flex_mask(
    item_ids: torch.Tensor,
    session_offsets: torch.Tensor,
) -> object | None:
    """Build one reusable block-sparse session-local mask for a packed training batch."""
    if create_block_mask is None or not item_ids.is_cuda:
        return None
    lengths = session_offsets[1:] - session_offsets[:-1]
    segment_ids = torch.repeat_interleave(
        torch.arange(lengths.numel(), device=item_ids.device),
        lengths.to(dtype=torch.long),
    )
    if segment_ids.numel() != item_ids.numel():
        raise ValueError("session_offsets must exactly cover item_ids")

    def mask_mod(
        _batch_index: torch.Tensor,
        _head_index: torch.Tensor,
        query_index: torch.Tensor,
        key_index: torch.Tensor,
    ) -> torch.Tensor:
        causal = query_index >= key_index
        same_session = segment_ids[query_index] == segment_ids[key_index]
        structural = (item_ids[key_index] == 0) | (item_ids[query_index] == item_ids[key_index])
        return causal & same_session & structural

    token_count = item_ids.numel()
    return create_block_mask(
        mask_mod,
        B=1,
        H=None,
        Q_LEN=token_count,
        KV_LEN=token_count,
        device=item_ids.device,
    )


def build_session_causal_flex_mask(
    item_ids: torch.Tensor,
    session_ids: torch.Tensor,
    sequence_offsets: torch.Tensor,
) -> object | None:
    """Build a session-causal mask for a packed batch."""
    if create_block_mask is None or not item_ids.is_cuda:
        return None
    lengths = sequence_offsets[1:] - sequence_offsets[:-1]
    sequence_ids = torch.repeat_interleave(
        torch.arange(lengths.numel(), device=item_ids.device),
        lengths.to(dtype=torch.long),
    )
    if sequence_ids.numel() != item_ids.numel() or session_ids.shape != item_ids.shape:
        raise ValueError("sequence_offsets, item_ids, and session_ids must cover the same tokens")

    def mask_mod(
        _batch_index: torch.Tensor,
        _head_index: torch.Tensor,
        query_index: torch.Tensor,
        key_index: torch.Tensor,
    ) -> torch.Tensor:
        causal = query_index >= key_index
        same_sequence = sequence_ids[query_index] == sequence_ids[key_index]
        previous_session = session_ids[query_index] > session_ids[key_index]
        same_session = session_ids[query_index] == session_ids[key_index]
        same_session_visible = (item_ids[key_index] == 0) | (item_ids[query_index] == item_ids[key_index])
        return causal & same_sequence & (previous_session | (same_session & same_session_visible))

    token_count = item_ids.numel()
    return create_block_mask(
        mask_mod,
        B=1,
        H=None,
        Q_LEN=token_count,
        KV_LEN=token_count,
        device=item_ids.device,
    )


def session_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask: object,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Fused structured-attention path with a block-sparse PyTorch/Triton backward."""
    if _compiled_flex_attention is None:
        raise RuntimeError("FlexAttention is unavailable in this PyTorch installation")
    mixed = _compiled_flex_attention(
        query.transpose(0, 1).unsqueeze(0),
        key.transpose(0, 1).unsqueeze(0),
        value.transpose(0, 1).unsqueeze(0),
        block_mask=block_mask,
        scale=scale,
    )
    return mixed.squeeze(0).transpose(0, 1)


class HeadRMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        normalized = tensor.float() * torch.rsqrt(
            tensor.float().square().mean(dim=-1, keepdim=True) + self.epsilon
        )
        return (normalized * self.weight.float()).to(tensor.dtype)


def segmented_causal_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sequence_offsets: torch.Tensor,
) -> torch.Tensor:
    outputs = []
    boundaries = sequence_offsets.detach().cpu().tolist()
    for start, end in pairwise(boundaries):
        q = query[start:end].transpose(0, 1).unsqueeze(0)
        k = key[start:end].transpose(0, 1).unsqueeze(0)
        v = value[start:end].transpose(0, 1).unsqueeze(0)
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        outputs.append(output.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0) if outputs else torch.empty_like(query)


class SegmentedSelfAttention(nn.Module):
    def __init__(
        self,
        model_width: int,
        heads: int,
        head_width: int,
        *,
        pattern: Literal["session", "session_causal", "causal"],
        rope_base: float = 1_000_000.0,
        qk_normalization: bool = True,
        use_rope: bool = True,
        rope_position_mode: Literal["document_aligned", "sequential"] = "document_aligned",
        use_triton: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.head_width = head_width
        self.pattern = pattern
        self.rope_base = rope_base
        self.use_rope = use_rope
        self.rope_position_mode = rope_position_mode
        self.use_triton = use_triton
        projection_width = heads * head_width
        self.query_projection = nn.Linear(model_width, projection_width, bias=False)
        self.key_projection = nn.Linear(model_width, projection_width, bias=False)
        self.value_projection = nn.Linear(model_width, projection_width, bias=False)
        self.output_projection = nn.Linear(projection_width, model_width, bias=False)
        self.query_norm = HeadRMSNorm(head_width) if qk_normalization else nn.Identity()
        self.key_norm = HeadRMSNorm(head_width) if qk_normalization else nn.Identity()
        self.output_dropout = nn.Dropout(dropout)
        self.last_attention: list[torch.Tensor] | None = None

    def _project_qkv(
        self,
        hidden: torch.Tensor,
        rope_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = hidden.shape[0]
        query = self.query_projection(hidden).view(tokens, self.heads, self.head_width)
        key = self.key_projection(hidden).view(tokens, self.heads, self.head_width)
        value = self.value_projection(hidden).view(tokens, self.heads, self.head_width)
        query = self.query_norm(query)
        key = self.key_norm(key)
        if self.use_rope:
            query = apply_rotary(query, rope_positions, base=self.rope_base)
            key = apply_rotary(key, rope_positions, base=self.rope_base)
        return query, key, value

    def _resolve_rope_positions(
        self,
        rope_positions: torch.Tensor,
        sequence_offsets: torch.Tensor,
    ) -> torch.Tensor:
        if self.rope_position_mode == "document_aligned":
            return rope_positions
        resolved = torch.empty_like(rope_positions)
        for start, end in pairwise(sequence_offsets.detach().cpu().tolist()):
            resolved[start:end] = torch.arange(end - start, device=resolved.device)
        return resolved

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        rope_positions: torch.Tensor,
        sequence_offsets: torch.Tensor,
        session_offsets: torch.Tensor,
        session_ids: torch.Tensor,
        item_ids: torch.Tensor,
        capture_attention: bool = False,
        session_block_mask: object | None = None,
    ) -> torch.Tensor:
        tokens = hidden.shape[0]
        rope_positions = self._resolve_rope_positions(rope_positions, sequence_offsets)
        query, key, value = self._project_qkv(hidden, rope_positions)

        if self.pattern in {"session", "session_causal"}:
            if capture_attention:
                offsets = session_offsets if self.pattern == "session" else sequence_offsets
                mixed, weights = segmented_attention_reference(
                    query,
                    key,
                    value,
                    offsets,
                    item_ids=item_ids,
                    session_ids=session_ids,
                    pattern=self.pattern,
                    return_weights=True,
                )
                self.last_attention = weights
            elif session_block_mask is not None:
                mixed = session_flex_attention(
                    query,
                    key,
                    value,
                    session_block_mask,
                )
                self.last_attention = None
            elif self.pattern == "session_causal":
                mixed = segmented_attention_reference(
                    query,
                    key,
                    value,
                    sequence_offsets,
                    item_ids=item_ids,
                    session_ids=session_ids,
                    pattern="session_causal",
                )
                self.last_attention = None
            else:
                mixed = session_flash_attention(
                    query,
                    key,
                    value,
                    item_ids,
                    session_offsets,
                    use_triton=self.use_triton,
                )
                self.last_attention = None
        else:
            if capture_attention:
                result = segmented_attention_reference(
                    query,
                    key,
                    value,
                    sequence_offsets,
                    pattern="causal",
                    return_weights=True,
                )
                mixed, self.last_attention = result
            else:
                mixed = segmented_causal_sdpa(query, key, value, sequence_offsets)
                self.last_attention = None
        projected = self.output_projection(mixed.reshape(tokens, -1))
        return self.output_dropout(projected)

    def prefill_cache(
        self,
        hidden: torch.Tensor,
        *,
        rope_positions: torch.Tensor,
        item_ids: torch.Tensor,
        session_ids: torch.Tensor,
        sequence_offsets: torch.Tensor,
        current_session_starts: list[int],
    ) -> SessionAttentionCache:
        """Cache the final session for local attention or the full prefix for causal attention."""
        if self.pattern not in {"session", "session_causal", "causal"}:
            raise ValueError("Unsupported attention pattern for cached decoding")
        rope_positions = self._resolve_rope_positions(rope_positions, sequence_offsets)
        query, key, value = self._project_qkv(hidden, rope_positions)
        sequence_starts = sequence_offsets[:-1].detach().cpu().tolist()
        sequence_ends = sequence_offsets[1:].detach().cpu().tolist()
        starts = current_session_starts if self.pattern == "session" else sequence_starts
        lengths = [end - start for start, end in zip(starts, sequence_ends, strict=True)]
        maximum_length = max(lengths)
        batch_size = len(lengths)
        query_cache_length = maximum_length if self.pattern == "session" else 0
        cached_query = query.new_zeros((batch_size, query_cache_length, self.heads, self.head_width))
        cached_key = key.new_zeros((batch_size, maximum_length, self.heads, self.head_width))
        cached_value = value.new_zeros((batch_size, maximum_length, self.heads, self.head_width))
        cached_item_ids = item_ids.new_full((batch_size, maximum_length), -1)
        cached_session_ids = session_ids.new_full((batch_size, maximum_length), -1)
        for batch_index, (start, end) in enumerate(zip(starts, sequence_ends, strict=True)):
            length = end - start
            if self.pattern == "session":
                cached_query[batch_index, :length] = query[start:end]
            cached_key[batch_index, :length] = key[start:end]
            cached_value[batch_index, :length] = value[start:end]
            cached_item_ids[batch_index, :length] = item_ids[start:end]
            cached_session_ids[batch_index, :length] = session_ids[start:end]
        return SessionAttentionCache(
            query=cached_query,
            key=cached_key,
            value=cached_value,
            item_ids=cached_item_ids,
            session_ids=cached_session_ids,
        )

    def decode_cached(
        self,
        hidden: torch.Tensor,
        *,
        rope_positions: torch.Tensor,
        cache: SessionAttentionCache,
    ) -> tuple[torch.Tensor, SessionAttentionCache]:
        """Decode one token per beam using the cached attention prefix."""
        if self.pattern not in {"session", "session_causal", "causal"} or hidden.ndim != 2:
            raise ValueError("Cached attention decoding expects packed [beams, width] input")
        if self.rope_position_mode == "sequential":
            rope_positions = (cache.item_ids >= 0).sum(dim=1)
        query, key, value = self._project_qkv(hidden, rope_positions)
        current_session_ids = cache.session_ids.max(dim=1).values
        updated_query = (
            torch.cat((cache.query, query[:, None]), dim=1) if self.pattern == "session" else cache.query
        )
        updated = SessionAttentionCache(
            query=updated_query,
            key=torch.cat((cache.key, key[:, None]), dim=1),
            value=torch.cat((cache.value, value[:, None]), dim=1),
            item_ids=torch.cat(
                (
                    cache.item_ids,
                    torch.ones((hidden.shape[0], 1), dtype=torch.long, device=hidden.device),
                ),
                dim=1,
            ),
            session_ids=torch.cat((cache.session_ids, current_session_ids[:, None]), dim=1),
        )
        if self.pattern == "causal":
            visible = updated.item_ids >= 0
            mixed = F.scaled_dot_product_attention(
                query.unsqueeze(2),
                updated.key.permute(0, 2, 1, 3),
                updated.value.permute(0, 2, 1, 3),
                attn_mask=visible[:, None, None, :],
            ).squeeze(2)
            projected = self.output_projection(mixed.reshape(hidden.shape[0], -1))
            return self.output_dropout(projected), updated
        if self.pattern == "session_causal":
            key_session_ids = updated.session_ids
            key_item_ids = updated.item_ids
            visible = (key_item_ids >= 0) & (
                (key_session_ids < current_session_ids[:, None])
                | (
                    (key_session_ids == current_session_ids[:, None])
                    & ((key_item_ids == 0) | (key_item_ids == 1))
                )
            )
            mixed = F.scaled_dot_product_attention(
                query.unsqueeze(2),
                updated.key.permute(0, 2, 1, 3),
                updated.value.permute(0, 2, 1, 3),
                attn_mask=visible[:, None, None, :],
            ).squeeze(2)
            projected = self.output_projection(mixed.reshape(hidden.shape[0], -1))
            return self.output_dropout(projected), updated
        sequence_length = updated.key.shape[1]
        offsets = torch.arange(
            0,
            (hidden.shape[0] + 1) * sequence_length,
            sequence_length,
            dtype=torch.int32,
            device=hidden.device,
        )
        mixed = session_flash_attention(
            updated.query.flatten(0, 1),
            updated.key.flatten(0, 1),
            updated.value.flatten(0, 1),
            updated.item_ids.flatten(),
            offsets,
            use_triton=self.use_triton,
        ).view(hidden.shape[0], sequence_length, self.heads, self.head_width)
        projected = self.output_projection(mixed[:, -1].reshape(hidden.shape[0], -1))
        return self.output_dropout(projected), updated
