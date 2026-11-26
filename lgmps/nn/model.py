from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from lgmps.configuration import ModelSettings
from lgmps.data.assembler import PackedBatch, TokenNamespace

from .attention import (
    SegmentedSelfAttention,
    SessionAttentionCache,
    build_session_causal_flex_mask,
    build_session_local_flex_mask,
)
from .delta_net import build_delta_net
from .feedforward import (
    AuxiliaryLossFreeMoE,
    CategoryFFN,
    CategoryMoE,
    RoutingSnapshot,
    SwiGLU,
)
from .gated_delta import GateCapture, build_gated_delta


@dataclass(slots=True)
class ModelOutput:
    hidden_states: torch.Tensor
    level_logits: list[torch.Tensor]
    loss: torch.Tensor | None
    level_losses: list[torch.Tensor]
    label_count: int
    level_label_counts: list[int]


def _index_cached_value(value: Any, indices: torch.Tensor) -> Any:
    if isinstance(value, dict):
        return {key: _index_cached_value(item, indices) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_index_cached_value(item, indices) for item in value)
    if isinstance(value, list):
        return [_index_cached_value(item, indices) for item in value]
    if isinstance(value, torch.Tensor):
        return value.index_select(0, indices)
    return value


def _index_fla_cache(cache: object, indices: torch.Tensor) -> object:
    from fla.models.utils import Cache

    states = tuple(_index_cached_value(state, indices) for state in cache.to_legacy_cache())
    return Cache.from_legacy_cache(states, seen_tokens=cache.get_seq_length())


@dataclass(slots=True)
class GenerationState:
    history_cache: object | None
    local_caches: list[SessionAttentionCache | None]

    def index_select(self, indices: torch.Tensor) -> GenerationState:
        return GenerationState(
            history_cache=(
                None if self.history_cache is None else _index_fla_cache(self.history_cache, indices)
            ),
            local_caches=[
                None if cache is None else cache.index_select(indices) for cache in self.local_caches
            ],
        )


class LGMPSBlock(nn.Module):
    def __init__(self, settings: ModelSettings, layer_index: int) -> None:
        super().__init__()
        requested_pattern = settings.attention_pattern
        self.pattern = (
            "full_transformer"
            if requested_pattern == "mix_3_1" and (layer_index + 1) % 4 == 0
            else "gated_deltanet"
            if requested_pattern == "mix_3_1"
            else requested_pattern
        )
        self.fusion_bias = settings.fusion_bias
        self.input_norm = nn.RMSNorm(settings.width)
        self.feedforward_norm = nn.RMSNorm(settings.width)
        self.dropout = nn.Dropout(settings.dropout)
        if self.pattern in {
            "full_transformer",
            "session_transformer",
            "session_causal_transformer",
        }:
            attention_pattern = {
                "full_transformer": "causal",
                "session_transformer": "session",
                "session_causal_transformer": "session_causal",
            }[self.pattern]
            self.attention = SegmentedSelfAttention(
                settings.width,
                settings.attention_heads,
                settings.head_width,
                pattern=attention_pattern,
                rope_base=settings.rope_base,
                qk_normalization=settings.qk_normalization,
                use_rope=settings.use_rope,
                rope_position_mode=settings.rope_position_mode,
                use_triton=settings.use_triton_session_attention,
                dropout=settings.dropout,
            )
            self.history = None
            self.local_attention = None
        elif self.pattern == "gated_deltanet":
            self.attention = None
            self.local_attention = None
            self.history = build_gated_delta(
                settings.history_backend,
                settings.width,
                head_dim=settings.head_width,
                num_heads=settings.history_heads,
                expand_v=settings.history_value_expansion,
                layer_idx=(
                    layer_index - (layer_index + 1) // 4 if requested_pattern == "mix_3_1" else layer_index
                ),
            )
        else:
            self.attention = None
            self.local_attention = SegmentedSelfAttention(
                settings.width,
                settings.local_heads,
                settings.head_width,
                pattern="session",
                rope_base=settings.rope_base,
                qk_normalization=settings.qk_normalization,
                use_rope=settings.use_rope,
                rope_position_mode=settings.rope_position_mode,
                use_triton=settings.use_triton_session_attention,
                dropout=settings.dropout,
            )
            if self.pattern == "para_chan_film_deltanet":
                self.history = build_delta_net(
                    settings.width,
                    head_dim=settings.head_width,
                    num_heads=settings.history_heads,
                    expand_v=settings.history_value_expansion,
                    layer_idx=layer_index,
                    mode=settings.delta_mode,
                    use_output_gate=settings.delta_output_gate,
                )
            else:
                self.history = build_gated_delta(
                    settings.history_backend,
                    settings.width,
                    head_dim=settings.head_width,
                    num_heads=settings.history_heads,
                    expand_v=settings.history_value_expansion,
                    layer_idx=layer_index,
                    output_multiplier=2,
                    output_bias=True,
                    zero_output=True,
                )
        if settings.feedforward == "category_moe":
            self.feedforward = CategoryMoE(
                settings.width,
                settings.expert_width,
                settings.routed_experts,
                settings.routed_top_k,
                settings.dropout,
            )
        elif settings.feedforward in {"moe", "pcmoe"}:
            self.feedforward = AuxiliaryLossFreeMoE(
                settings.width,
                settings.expert_width,
                settings.routed_experts,
                settings.routed_top_k,
                shared_experts=settings.shared_experts,
                category_shared_experts=settings.category_shared_experts,
                normalize_topk_probability=settings.moe_normalize_topk_probability,
                balance_momentum=settings.moe_balance_momentum,
                balance_bias_lr=settings.moe_balance_bias_lr,
                balance_bias_max=settings.moe_balance_bias_max,
                dropout=settings.dropout,
            )
        elif settings.feedforward == "category_ffn":
            self.feedforward = CategoryFFN(
                settings.width,
                settings.feedforward_width,
                settings.dropout,
            )
        else:
            self.feedforward = SwiGLU(settings.width, settings.feedforward_width, settings.dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        batch: PackedBatch,
        *,
        capture_attention: bool = False,
        gate_capture: GateCapture | None = None,
        session_block_mask: object | None = None,
    ) -> torch.Tensor:
        normalized = self.input_norm(hidden)
        if self.pattern in {
            "full_transformer",
            "session_transformer",
            "session_causal_transformer",
        }:
            attention_output = self.attention(
                normalized,
                rope_positions=batch.rope_positions,
                sequence_offsets=batch.sequence_offsets,
                session_offsets=batch.session_offsets,
                session_ids=batch.session_ids,
                item_ids=batch.item_ids,
                capture_attention=capture_attention,
                session_block_mask=session_block_mask,
            )
        elif self.pattern == "gated_deltanet":
            attention_output = self.history(
                normalized,
                sequence_offsets=batch.sequence_offsets,
                gate_capture=gate_capture,
            )
        else:
            history_output = self.history(
                normalized,
                sequence_offsets=batch.sequence_offsets,
                gate_capture=gate_capture,
            )
            local_output = self.local_attention(
                normalized,
                rope_positions=batch.rope_positions,
                sequence_offsets=batch.sequence_offsets,
                session_offsets=batch.session_offsets,
                session_ids=batch.session_ids,
                item_ids=batch.item_ids,
                capture_attention=capture_attention,
                session_block_mask=session_block_mask,
            )
            if self.pattern == "para_chan_film_deltanet":
                attention_output = local_output * (1.0 + history_output)
            else:
                gamma, beta = history_output.chunk(2, dim=-1)
                attention_output = local_output * (1.0 + gamma)
                if self.fusion_bias:
                    attention_output = attention_output + beta
        hidden = hidden + self.dropout(attention_output)
        feedforward_input = self.feedforward_norm(hidden)
        if isinstance(self.feedforward, (AuxiliaryLossFreeMoE, CategoryFFN, CategoryMoE)):
            feedforward_output = self.feedforward(feedforward_input, batch.categories)
        else:
            feedforward_output = self.feedforward(feedforward_input)
        return hidden + self.dropout(feedforward_output)

    def _feedforward_step(self, hidden: torch.Tensor, categories: torch.Tensor) -> torch.Tensor:
        feedforward_input = self.feedforward_norm(hidden)
        if isinstance(self.feedforward, (AuxiliaryLossFreeMoE, CategoryFFN, CategoryMoE)):
            feedforward_output = self.feedforward(feedforward_input, categories)
        else:
            feedforward_output = self.feedforward(feedforward_input)
        return hidden + self.dropout(feedforward_output)

    def prefill_generation(
        self,
        hidden: torch.Tensor,
        batch: PackedBatch,
        history_cache: object | None,
        current_session_starts: list[int],
        session_block_mask: object | None = None,
    ) -> tuple[torch.Tensor, SessionAttentionCache | None]:
        normalized = self.input_norm(hidden)
        if self.pattern in {"session_causal_transformer", "full_transformer"} and self.attention is not None:
            attention_output = self.attention(
                normalized,
                rope_positions=batch.rope_positions,
                sequence_offsets=batch.sequence_offsets,
                session_offsets=batch.session_offsets,
                session_ids=batch.session_ids,
                item_ids=batch.item_ids,
                session_block_mask=session_block_mask,
            )
            attention_cache = self.attention.prefill_cache(
                normalized,
                rope_positions=batch.rope_positions,
                item_ids=batch.item_ids,
                session_ids=batch.session_ids,
                sequence_offsets=batch.sequence_offsets,
                current_session_starts=current_session_starts,
            )
            return (
                self._feedforward_step(hidden + self.dropout(attention_output), batch.categories),
                attention_cache,
            )
        if self.pattern == "gated_deltanet" and self.history is not None and history_cache is not None:
            attention_output = self.history.prefill_cached(
                normalized,
                batch.sequence_offsets,
                history_cache,
            )
            return self._feedforward_step(hidden + self.dropout(attention_output), batch.categories), None
        if (
            self.pattern not in {"para_chan_film", "para_chan_film_deltanet"}
            or self.local_attention is None
            or self.history is None
            or history_cache is None
        ):
            raise ValueError("Cached generation is unavailable for this attention pattern")
        history_output = self.history.prefill_cached(
            normalized,
            batch.sequence_offsets,
            history_cache,
        )
        local_output = self.local_attention(
            normalized,
            rope_positions=batch.rope_positions,
            sequence_offsets=batch.sequence_offsets,
            session_offsets=batch.session_offsets,
            session_ids=batch.session_ids,
            item_ids=batch.item_ids,
            session_block_mask=session_block_mask,
        )
        local_cache = self.local_attention.prefill_cache(
            normalized,
            rope_positions=batch.rope_positions,
            item_ids=batch.item_ids,
            session_ids=batch.session_ids,
            sequence_offsets=batch.sequence_offsets,
            current_session_starts=current_session_starts,
        )
        if self.pattern == "para_chan_film_deltanet":
            attention_output = local_output * (1.0 + history_output)
        else:
            gamma, beta = history_output.chunk(2, dim=-1)
            attention_output = local_output * (1.0 + gamma)
            if self.fusion_bias:
                attention_output = attention_output + beta
        return self._feedforward_step(hidden + self.dropout(attention_output), batch.categories), local_cache

    def decode_generation(
        self,
        hidden: torch.Tensor,
        *,
        categories: torch.Tensor,
        rope_positions: torch.Tensor,
        history_cache: object | None,
        local_cache: SessionAttentionCache | None,
    ) -> tuple[torch.Tensor, SessionAttentionCache | None]:
        normalized = self.input_norm(hidden)
        if self.pattern in {"session_causal_transformer", "full_transformer"} and self.attention is not None:
            if local_cache is None:
                raise ValueError("Full-attention cached decoding requires a KV cache")
            attention_output, updated_attention_cache = self.attention.decode_cached(
                normalized,
                rope_positions=rope_positions,
                cache=local_cache,
            )
            hidden = self._feedforward_step(hidden + self.dropout(attention_output), categories)
            return hidden, updated_attention_cache
        if self.pattern == "gated_deltanet" and self.history is not None and history_cache is not None:
            attention_output = self.history.decode_cached(normalized, history_cache)
            hidden = self._feedforward_step(hidden + self.dropout(attention_output), categories)
            return hidden, None
        if (
            self.pattern not in {"para_chan_film", "para_chan_film_deltanet"}
            or self.local_attention is None
            or self.history is None
            or history_cache is None
        ):
            raise ValueError("Cached generation is unavailable for this attention pattern")
        if local_cache is None:
            raise ValueError("Parallel-channel cached decoding requires a local-attention cache")
        history_output = self.history.decode_cached(normalized, history_cache)
        local_output, updated_local_cache = self.local_attention.decode_cached(
            normalized,
            rope_positions=rope_positions,
            cache=local_cache,
        )
        if self.pattern == "para_chan_film_deltanet":
            attention_output = local_output * (1.0 + history_output)
        else:
            gamma, beta = history_output.chunk(2, dim=-1)
            attention_output = local_output * (1.0 + gamma)
            if self.fusion_bias:
                attention_output = attention_output + beta
        hidden = self._feedforward_step(hidden + self.dropout(attention_output), categories)
        return hidden, updated_local_cache


class LGMPSModel(nn.Module):
    def __init__(self, settings: ModelSettings) -> None:
        super().__init__()
        settings.validate()
        self.settings = settings
        self.semantic_offsets = [0]
        for size in settings.semantic_levels:
            self.semantic_offsets.append(self.semantic_offsets[-1] + size)
        total_semantic = self.semantic_offsets[-1]
        self.query_embedding = nn.Embedding(settings.query_vocabulary + 1, settings.width, padding_idx=0)
        self.semantic_embedding = nn.Embedding(total_semantic + 1, settings.width, padding_idx=0)
        self.special_embedding = nn.Embedding(3, settings.width)
        self.category_embedding = nn.Embedding(2, settings.width)
        self.embedding_norm = nn.RMSNorm(settings.width)
        self.embedding_dropout = nn.Dropout(settings.dropout)
        self.blocks = nn.ModuleList(LGMPSBlock(settings, index) for index in range(settings.layers))
        self.final_norm = nn.RMSNorm(settings.width)
        self.level_heads = None
        if not settings.tie_semantic_embedding:
            self.level_heads = nn.ModuleList(
                nn.Linear(settings.width, size, bias=False) for size in settings.semantic_levels
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (
            self.query_embedding,
            self.semantic_embedding,
            self.special_embedding,
            self.category_embedding,
        ):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.query_embedding.weight[0].zero_()
            self.semantic_embedding.weight[0].zero_()

    def embed(self, batch: PackedBatch) -> torch.Tensor:
        hidden = self.category_embedding(batch.categories)
        for namespace, embedding in (
            (TokenNamespace.QUERY, self.query_embedding),
            (TokenNamespace.SEMANTIC, self.semantic_embedding),
            (TokenNamespace.SPECIAL, self.special_embedding),
        ):
            selected = batch.namespaces == int(namespace)
            if selected.any():
                hidden[selected] = hidden[selected] + embedding(batch.token_ids[selected])
        return self.embedding_dropout(self.embedding_norm(hidden))

    def _level_logits(self, hidden: torch.Tensor, level: int) -> torch.Tensor:
        if self.level_heads is not None:
            return self.level_heads[level](hidden)
        # Row zero is padding. Global SID (offset + 1) is local class zero.
        start = self.semantic_offsets[level] + 1
        end = self.semantic_offsets[level + 1] + 1
        return F.linear(hidden, self.semantic_embedding.weight[start:end])

    def losses_for_labels(
        self,
        logits: list[torch.Tensor],
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor | None, list[torch.Tensor], int, list[int]]:
        losses: list[torch.Tensor] = []
        counts: list[int] = []
        for level, level_logits in enumerate(logits):
            lower = self.semantic_offsets[level] + 1
            upper = self.semantic_offsets[level + 1]
            selected = (labels >= lower) & (labels <= upper)
            count = int(selected.sum().item())
            counts.append(count)
            if count:
                target = labels[selected] - lower
                losses.append(F.cross_entropy(level_logits[selected].float(), target))
            else:
                losses.append(level_logits.sum() * 0.0)
        active = [loss for loss, count in zip(losses, counts, strict=True) if count]
        return (torch.stack(active).mean() if active else None), losses, sum(counts), counts

    def forward(
        self,
        batch: PackedBatch,
        *,
        label_split: str | None = "train",
        capture_attention: bool = False,
        gate_capture: GateCapture | None = None,
    ) -> ModelOutput:
        hidden = self.embed(batch)
        session_block_mask = None
        if (
            self.training
            and not capture_attention
            and self.settings.use_flex_training_attention
            and self.settings.attention_pattern
            in {"session_transformer", "para_chan_film", "para_chan_film_deltanet"}
        ):
            session_block_mask = build_session_local_flex_mask(batch.item_ids, batch.session_offsets)
        elif (
            self.training
            and not capture_attention
            and self.settings.use_flex_training_attention
            and self.settings.attention_pattern == "session_causal_transformer"
        ):
            session_block_mask = build_session_causal_flex_mask(
                batch.item_ids,
                batch.session_ids,
                batch.sequence_offsets,
            )
        for block in self.blocks:
            hidden = block(
                hidden,
                batch,
                capture_attention=capture_attention,
                gate_capture=gate_capture,
                session_block_mask=session_block_mask,
            )
        hidden = self.final_norm(hidden)
        logits = [self._level_logits(hidden, level) for level in range(len(self.semantic_offsets) - 1)]
        if label_split is None:
            loss, losses, count, counts = None, [], 0, [0] * len(logits)
        else:
            labels = batch.train_labels if label_split == "train" else batch.validation_labels
            loss, losses, count, counts = self.losses_for_labels(logits, labels)
        return ModelOutput(hidden, logits, loss, losses, count, counts)

    @property
    def supports_generation_cache(self) -> bool:
        return self.settings.attention_pattern == "session_causal_transformer" or (
            self.settings.attention_pattern in {"mix_3_1", "para_chan_film", "para_chan_film_deltanet"}
            and self.settings.history_backend == "fla"
        )

    def prefill_generation(
        self,
        batch: PackedBatch,
        *,
        level: int = 0,
    ) -> tuple[torch.Tensor, GenerationState]:
        if not self.supports_generation_cache:
            raise ValueError("Cached generation is unavailable for this architecture")

        history_cache = None
        if self.settings.attention_pattern in {
            "mix_3_1",
            "para_chan_film",
            "para_chan_film_deltanet",
        }:
            from fla.models.utils import Cache

            history_cache = Cache()
        sequence_boundaries = batch.sequence_offsets.detach().cpu().tolist()
        session_ids = batch.session_ids.detach().cpu().tolist()
        current_session_starts: list[int] = []
        for sequence_start, sequence_end in pairwise(sequence_boundaries):
            final_session_id = session_ids[sequence_end - 1]
            current_start = sequence_end - 1
            while current_start > sequence_start and session_ids[current_start - 1] == final_session_id:
                current_start -= 1
            current_session_starts.append(current_start)
        hidden = self.embed(batch)
        local_caches: list[SessionAttentionCache | None] = []
        session_block_mask = None
        if (
            self.settings.use_flex_training_attention
            and self.settings.attention_pattern == "session_causal_transformer"
        ):
            session_block_mask = build_session_causal_flex_mask(
                batch.item_ids,
                batch.session_ids,
                batch.sequence_offsets,
            )
        for block in self.blocks:
            hidden, local_cache = block.prefill_generation(
                hidden,
                batch,
                history_cache,
                current_session_starts,
                session_block_mask,
            )
            local_caches.append(local_cache)
        final_positions = batch.sequence_offsets[1:].long() - 1
        final_hidden = self.final_norm(hidden[final_positions])
        return self._level_logits(final_hidden, level), GenerationState(history_cache, local_caches)

    def decode_generation(
        self,
        token_ids: torch.Tensor,
        *,
        rope_positions: torch.Tensor,
        state: GenerationState,
        level: int,
    ) -> tuple[torch.Tensor, GenerationState]:
        if not self.supports_generation_cache or token_ids.ndim != 1:
            raise ValueError("Cached generation expects one semantic token per beam")
        categories = torch.ones_like(token_ids)
        hidden = self.category_embedding(categories) + self.semantic_embedding(token_ids)
        hidden = self.embedding_dropout(self.embedding_norm(hidden))
        updated_local: list[SessionAttentionCache] = []
        for block, local_cache in zip(self.blocks, state.local_caches, strict=True):
            hidden, local_cache = block.decode_generation(
                hidden,
                categories=categories,
                rope_positions=rope_positions,
                history_cache=state.history_cache,
                local_cache=local_cache,
            )
            updated_local.append(local_cache)
        logits = self._level_logits(self.final_norm(hidden), level)
        return logits, GenerationState(state.history_cache, updated_local)

    def routing_snapshots(self) -> list[RoutingSnapshot | None]:
        snapshots: list[RoutingSnapshot | None] = []
        for block in self.blocks:
            snapshots.append(
                block.feedforward.last_routing
                if isinstance(block.feedforward, (AuxiliaryLossFreeMoE, CategoryMoE))
                else None
            )
        return snapshots

    def set_moe_intervention(self, intervention: str) -> None:
        modules = [
            block.feedforward for block in self.blocks if isinstance(block.feedforward, AuxiliaryLossFreeMoE)
        ]
        if not modules:
            raise ValueError("The configured model has no auxiliary-loss-free MoE layers")
        for module in modules:
            module.set_intervention(intervention)

    def enable_moe_contribution_capture(self, enabled: bool = True) -> None:
        modules = [
            block.feedforward for block in self.blocks if isinstance(block.feedforward, AuxiliaryLossFreeMoE)
        ]
        if not modules:
            raise ValueError("The configured model has no auxiliary-loss-free MoE layers")
        for module in modules:
            module.enable_contribution_capture(enabled)

    def attention_captures(self) -> list[list[torch.Tensor] | None]:
        captures: list[list[torch.Tensor] | None] = []
        for block in self.blocks:
            attention = block.local_attention if block.local_attention is not None else block.attention
            captures.append(None if attention is None else attention.last_attention)
        return captures
