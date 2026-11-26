from __future__ import annotations

import torch

from .masking import segmented_attention_reference

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised by CPU-only installations
    triton = None
    tl = None


# Derived from the online-softmax structure in Triton's public fused-attention
# tutorial (MIT).
# Upstream commit: https://github.com/triton-lang/triton/blob/8bd8035559d4d2fa387149eecdf10d545d8a3d03/python/tutorials/06-fused-attention.py
if triton is not None:

    @triton.jit
    def _session_flash_forward(
        query,
        key,
        value,
        item_ids,
        segment_offsets,
        output,
        token_count: tl.constexpr,
        head_count: tl.constexpr,
        head_width: tl.constexpr,
        softmax_scale: tl.constexpr,
        block_rows: tl.constexpr,
        block_columns: tl.constexpr,
    ):
        row_block = tl.program_id(0)
        segment_head = tl.program_id(1)
        head_index = segment_head % head_count
        segment_index = segment_head // head_count

        segment_start = tl.load(segment_offsets + segment_index)
        segment_end = tl.load(segment_offsets + segment_index + 1)
        segment_length = segment_end - segment_start
        row_in_segment = row_block * block_rows + tl.arange(0, block_rows)
        column_in_head = tl.arange(0, head_width)
        row_valid = row_in_segment < segment_length
        absolute_rows = segment_start + row_in_segment

        query_offsets = absolute_rows[:, None] * head_count * head_width
        query_offsets += head_index * head_width + column_in_head[None, :]
        q = tl.load(query + query_offsets, mask=row_valid[:, None], other=0.0)
        query_item = tl.load(item_ids + absolute_rows, mask=row_valid, other=-1)

        row_max = tl.full([block_rows], -float("inf"), tl.float32)
        row_sum = tl.zeros([block_rows], tl.float32)
        accumulator = tl.zeros([block_rows, head_width], tl.float32)
        log2e_scale: tl.constexpr = softmax_scale * 1.4426950408889634

        for column_start in tl.range(0, segment_length, block_columns):
            column_in_segment = column_start + tl.arange(0, block_columns)
            column_valid = column_in_segment < segment_length
            absolute_columns = segment_start + column_in_segment
            key_offsets = absolute_columns[None, :] * head_count * head_width
            key_offsets += head_index * head_width + column_in_head[:, None]
            k = tl.load(key + key_offsets, mask=column_valid[None, :], other=0.0)

            scores = tl.dot(q, k) * log2e_scale
            key_item = tl.load(item_ids + absolute_columns, mask=column_valid, other=-2)
            causal = row_in_segment[:, None] >= column_in_segment[None, :]
            structural = (key_item[None, :] == 0) | (query_item[:, None] == key_item[None, :])
            visible = row_valid[:, None] & column_valid[None, :] & causal & structural
            # Extra programs may contain no real query rows. Give those rows a
            # harmless key so -inf - -inf never enters the online recurrence.
            harmless = (~row_valid[:, None]) & (column_in_segment[None, :] == 0)
            scores = tl.where(visible | harmless, scores, -float("inf"))

            block_max = tl.max(scores, axis=1)
            next_max = tl.maximum(row_max, block_max)
            correction = tl.exp2(row_max - next_max)
            probabilities = tl.exp2(scores - next_max[:, None])
            next_sum = row_sum * correction + tl.sum(probabilities, axis=1)

            value_offsets = absolute_columns[:, None] * head_count * head_width
            value_offsets += head_index * head_width + column_in_head[None, :]
            v = tl.load(value + value_offsets, mask=column_valid[:, None], other=0.0)
            accumulator *= correction[:, None]
            accumulator += tl.dot(probabilities.to(v.dtype), v)
            row_max = next_max
            row_sum = next_sum

        normalized = accumulator / row_sum[:, None]
        output_offsets = absolute_rows[:, None] * head_count * head_width
        output_offsets += head_index * head_width + column_in_head[None, :]
        tl.store(output + output_offsets, normalized, mask=row_valid[:, None])


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    item_ids: torch.Tensor,
    segment_offsets: torch.Tensor,
) -> None:
    if query.ndim != 3 or key.shape != query.shape or value.shape != query.shape:
        raise ValueError("query, key, and value must share [tokens, heads, head_width]")
    if item_ids.shape != (query.shape[0],):
        raise ValueError("item_ids must have one value per token")
    if segment_offsets.ndim != 1 or segment_offsets.numel() < 2:
        raise ValueError("segment_offsets must have shape [segments + 1]")
    if int(segment_offsets[0]) != 0 or int(segment_offsets[-1]) != query.shape[0]:
        raise ValueError("segment_offsets must exactly cover the token axis")


def _launch_triton(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    item_ids: torch.Tensor,
    segment_offsets: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not query.is_cuda:
        raise RuntimeError("The Triton session kernel requires CUDA tensors")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("The Triton session kernel supports float16 and bfloat16")
    if query.shape[-1] not in (16, 32, 64, 128):
        raise ValueError("The Triton session kernel supports head widths 16, 32, 64, and 128")
    query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
    item_ids = item_ids.to(device=query.device, dtype=torch.int32).contiguous()
    segment_offsets = segment_offsets.to(device=query.device, dtype=torch.int32).contiguous()
    lengths = segment_offsets[1:] - segment_offsets[:-1]
    maximum_segment = int(lengths.max().item())
    block_rows = 64
    block_columns = 64
    output = torch.empty_like(query)
    grid = (triton.cdiv(maximum_segment, block_rows), int(lengths.numel()) * query.shape[1])
    _session_flash_forward[grid](
        query,
        key,
        value,
        item_ids,
        segment_offsets,
        output,
        token_count=query.shape[0],
        head_count=query.shape[1],
        head_width=query.shape[2],
        softmax_scale=scale,
        block_rows=block_rows,
        block_columns=block_columns,
        num_warps=4,
        num_stages=2,
    )
    return output


class _SessionFlashFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        item_ids: torch.Tensor,
        segment_offsets: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        _validate_inputs(query, key, value, item_ids, segment_offsets)
        output = _launch_triton(query, key, value, item_ids, segment_offsets, scale)
        ctx.save_for_backward(query, key, value, item_ids, segment_offsets)
        ctx.scale = scale
        return output

    @staticmethod
    def backward(ctx, output_gradient: torch.Tensor):
        query, key, value, item_ids, segment_offsets = ctx.saved_tensors
        needs = ctx.needs_input_grad[:3]
        with torch.enable_grad():
            q = query.detach().requires_grad_(needs[0])
            k = key.detach().requires_grad_(needs[1])
            v = value.detach().requires_grad_(needs[2])
            reference = segmented_attention_reference(
                q,
                k,
                v,
                segment_offsets,
                item_ids=item_ids,
                pattern="session",
                scale=ctx.scale,
            )
            requested = [tensor for tensor, needed in zip((q, k, v), needs, strict=True) if needed]
            computed = torch.autograd.grad(reference, requested, output_gradient, allow_unused=False)
        gradients = []
        cursor = 0
        for needed in needs:
            if needed:
                gradients.append(computed[cursor])
                cursor += 1
            else:
                gradients.append(None)
        return (*gradients, None, None, None)


def session_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    item_ids: torch.Tensor,
    segment_offsets: torch.Tensor,
    *,
    scale: float | None = None,
    use_triton: bool = True,
) -> torch.Tensor:
    """Session-local FlashAttention with a correct autograd fallback backward.

    The fused Triton forward uses online softmax and materializes no attention
    matrix. Its backward intentionally recomputes the public PyTorch oracle; this
    keeps training numerically transparent while the custom backward kernel is
    tested independently in future work.
    """
    _validate_inputs(query, key, value, item_ids, segment_offsets)
    resolved_scale = float(scale if scale is not None else query.shape[-1] ** -0.5)
    supported = (
        use_triton
        and triton is not None
        and query.is_cuda
        and query.dtype in (torch.float16, torch.bfloat16)
        and query.shape[-1] in (16, 32, 64, 128)
    )
    if supported:
        return _SessionFlashFunction.apply(query, key, value, item_ids, segment_offsets, resolved_scale)
    return segmented_attention_reference(
        query,
        key,
        value,
        segment_offsets,
        item_ids=item_ids,
        pattern="session",
        scale=resolved_scale,
    )
