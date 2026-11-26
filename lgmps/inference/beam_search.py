from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch

from lgmps.data.assembler import PackedBatch, SpecialToken, TokenNamespace
from lgmps.nn import LGMPSModel

from .packed_ops import append_token, combine_single_sequence_batches


@dataclass(slots=True)
class BeamResult:
    semantic_ids: list[tuple[int, ...]]
    scores: list[float]


@dataclass(slots=True)
class _Beam:
    semantic_ids: tuple[int, ...]
    score: float
    context: PackedBatch


@torch.inference_mode()
def generate_semantic_ids(
    model: LGMPSModel,
    prefix: PackedBatch,
    *,
    beam_size: int = 20,
    return_sequences: int = 20,
    temperature: float = 1.0,
    use_cache: bool = True,
) -> BeamResult:
    if prefix.batch_size != 1:
        raise ValueError("Beam generation currently accepts one user sequence at a time")
    if return_sequences > beam_size:
        raise ValueError("return_sequences cannot exceed beam_size")
    if use_cache and model.supports_generation_cache:
        return generate_semantic_ids_batch(
            model,
            [prefix],
            beam_size=beam_size,
            return_sequences=return_sequences,
            temperature=temperature,
            use_cache=True,
        )[0]
    marker_context = append_token(
        prefix,
        namespace=TokenNamespace.SPECIAL,
        token_id=int(SpecialToken.ITEM_START),
        item_id=1,
    )
    beams = [_Beam((), 0.0, marker_context)]
    for level, vocabulary_size in enumerate(model.settings.semantic_levels):
        packed = combine_single_sequence_batches([beam.context for beam in beams])
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if packed.token_ids.is_cuda
            else nullcontext()
        )
        with autocast:
            output = model(packed, label_split=None)
        final_positions = packed.sequence_offsets[1:].long() - 1
        log_probabilities = torch.log_softmax(
            output.level_logits[level][final_positions].float() / temperature, dim=-1
        )
        candidates: list[tuple[float, int, int]] = []
        per_beam = min(beam_size, vocabulary_size)
        values, indices = torch.topk(log_probabilities, k=per_beam, dim=-1)
        for beam_index, beam in enumerate(beams):
            for slot in range(per_beam):
                candidates.append(
                    (
                        beam.score + float(values[beam_index, slot]),
                        beam_index,
                        int(indices[beam_index, slot]),
                    )
                )
        candidates.sort(key=lambda value: value[0], reverse=True)
        next_beams: list[_Beam] = []
        global_offset = model.semantic_offsets[level] + 1
        for score, beam_index, local_token in candidates[:beam_size]:
            global_token = global_offset + local_token
            source = beams[beam_index]
            next_beams.append(
                _Beam(
                    semantic_ids=(*source.semantic_ids, global_token),
                    score=score,
                    context=append_token(
                        source.context,
                        namespace=TokenNamespace.SEMANTIC,
                        token_id=global_token,
                        item_id=1,
                    ),
                )
            )
        beams = next_beams
    selected = beams[:return_sequences]
    return BeamResult(
        semantic_ids=[beam.semantic_ids for beam in selected],
        scores=[beam.score for beam in selected],
    )


@torch.inference_mode()
def generate_semantic_ids_batch(
    model: LGMPSModel,
    prefixes: list[PackedBatch],
    *,
    beam_size: int = 20,
    return_sequences: int = 20,
    temperature: float = 1.0,
    use_cache: bool = True,
) -> list[BeamResult]:
    if not prefixes or any(prefix.batch_size != 1 for prefix in prefixes):
        raise ValueError("Batch generation expects non-empty single-sequence prefixes")
    if return_sequences > beam_size:
        raise ValueError("return_sequences cannot exceed beam_size")
    if not use_cache or not model.supports_generation_cache:
        return [
            generate_semantic_ids(
                model,
                prefix,
                beam_size=beam_size,
                return_sequences=return_sequences,
                temperature=temperature,
                use_cache=False,
            )
            for prefix in prefixes
        ]
    marker_context = combine_single_sequence_batches(
        [
            append_token(
                prefix,
                namespace=TokenNamespace.SPECIAL,
                token_id=int(SpecialToken.ITEM_START),
                item_id=1,
            )
            for prefix in prefixes
        ]
    )
    return _generate_semantic_ids_batch_cached(
        model,
        marker_context,
        beam_size=beam_size,
        return_sequences=return_sequences,
        temperature=temperature,
    )


def _autocast_for(batch: PackedBatch):
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16) if batch.token_ids.is_cuda else nullcontext()
    )


def _select_candidates(
    log_probabilities: torch.Tensor,
    scores: list[float],
    *,
    beam_size: int,
) -> list[tuple[float, int, int]]:
    per_beam = min(beam_size, log_probabilities.shape[-1])
    values, indices = torch.topk(log_probabilities, k=per_beam, dim=-1)
    candidates: list[tuple[float, int, int]] = []
    for beam_index, score in enumerate(scores):
        for slot in range(per_beam):
            candidates.append(
                (
                    score + float(values[beam_index, slot]),
                    beam_index,
                    int(indices[beam_index, slot]),
                )
            )
    candidates.sort(key=lambda value: value[0], reverse=True)
    return candidates[:beam_size]


def _generate_semantic_ids_batch_cached(
    model: LGMPSModel,
    marker_context: PackedBatch,
    *,
    beam_size: int,
    return_sequences: int,
    temperature: float,
) -> list[BeamResult]:
    """Beam decode several FiLM/GDN prefixes with independent per-request pruning."""
    with _autocast_for(marker_context):
        logits, state = model.prefill_generation(marker_context, level=0)
    log_probabilities = torch.log_softmax(logits.float() / temperature, dim=-1)
    device = marker_context.token_ids.device
    scores_by_sample: list[list[float]] = []
    semantic_ids_by_sample: list[list[tuple[int, ...]]] = []
    initial_state_indices: list[int] = []
    for sample_index in range(marker_context.batch_size):
        selected = _select_candidates(
            log_probabilities[sample_index : sample_index + 1],
            [0.0],
            beam_size=beam_size,
        )
        scores_by_sample.append([score for score, _, _ in selected])
        semantic_ids_by_sample.append(
            [(model.semantic_offsets[0] + 1 + local_token,) for _, _, local_token in selected]
        )
        initial_state_indices.extend([sample_index] * len(selected))
    state = state.index_select(torch.tensor(initial_state_indices, dtype=torch.long, device=device))
    current_tokens = torch.tensor(
        [tokens[-1] for sample in semantic_ids_by_sample for tokens in sample],
        dtype=torch.long,
        device=device,
    )
    marker_positions = marker_context.rope_positions[
        marker_context.sequence_offsets[1:].long() - 1
    ].repeat_interleave(beam_size)

    for level in range(1, len(model.settings.semantic_levels)):
        rope_positions = marker_positions + level
        with _autocast_for(marker_context):
            logits, state = model.decode_generation(
                current_tokens,
                rope_positions=rope_positions,
                state=state,
                level=level,
            )
        log_probabilities = torch.log_softmax(logits.float() / temperature, dim=-1)
        next_scores_by_sample: list[list[float]] = []
        next_semantic_ids_by_sample: list[list[tuple[int, ...]]] = []
        global_parents: list[int] = []
        cursor = 0
        for scores, semantic_ids in zip(scores_by_sample, semantic_ids_by_sample, strict=True):
            beam_count = len(scores)
            selected = _select_candidates(
                log_probabilities[cursor : cursor + beam_count],
                scores,
                beam_size=beam_size,
            )
            next_scores_by_sample.append([score for score, _, _ in selected])
            next_semantic_ids_by_sample.append(
                [
                    (
                        *semantic_ids[parent],
                        model.semantic_offsets[level] + 1 + local_token,
                    )
                    for _, parent, local_token in selected
                ]
            )
            global_parents.extend(cursor + parent for _, parent, _ in selected)
            cursor += beam_count
        scores_by_sample = next_scores_by_sample
        semantic_ids_by_sample = next_semantic_ids_by_sample
        if level + 1 < len(model.settings.semantic_levels):
            parents = torch.tensor(
                global_parents,
                dtype=torch.long,
                device=device,
            )
            state = state.index_select(parents)
            current_tokens = torch.tensor(
                [tokens[-1] for sample in semantic_ids_by_sample for tokens in sample],
                dtype=torch.long,
                device=device,
            )

    return [
        BeamResult(
            semantic_ids=semantic_ids[:return_sequences],
            scores=scores[:return_sequences],
        )
        for scores, semantic_ids in zip(scores_by_sample, semantic_ids_by_sample, strict=True)
    ]
