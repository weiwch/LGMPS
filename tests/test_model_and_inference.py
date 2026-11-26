from __future__ import annotations

import pytest
import torch

from lgmps.configuration import ModelSettings
from lgmps.data import SequenceAssembler, pack_sequences
from lgmps.inference import generate_semantic_ids, generate_semantic_ids_batch
from lgmps.nn import LGMPSModel


def _settings() -> ModelSettings:
    return ModelSettings(
        attention_pattern="para_chan_film",
        query_vocabulary=100,
        width=32,
        layers=1,
        head_width=8,
        attention_heads=4,
        local_heads=2,
        history_heads=2,
        history_value_expansion=2,
        feedforward_width=48,
        history_backend="reference",
        use_triton_session_attention=False,
    )


def test_film_model_loss_gradients_and_transformed_gates(toy_record):
    batch = pack_sequences([SequenceAssembler(maximum_tokens=128).assemble(toy_record)])
    settings = _settings()
    settings.use_flex_training_attention = False
    model = LGMPSModel(settings)
    history = model.blocks[0].history
    assert torch.count_nonzero(history.o_proj.weight) == 0
    captures = []
    output = model(batch, gate_capture=captures.append)
    assert output.loss is not None
    assert output.label_count == 3
    assert output.level_label_counts == [1, 1, 1]
    assert len(captures) == 1
    assert torch.all((captures[0]["alpha"] > 0) & (captures[0]["alpha"] <= 1))
    assert torch.all((captures[0]["beta"] > 0) & (captures[0]["beta"] < 1))
    output.loss.backward()
    assert torch.isfinite(model.query_embedding.weight.grad).all()


@pytest.mark.parametrize("tied", [True, False])
def test_semantic_output_weights_and_gradients(toy_record, tied):
    settings = _settings()
    if not tied:
        settings.tie_semantic_embedding = False
    model = LGMPSModel(settings)
    batch = pack_sequences([SequenceAssembler(maximum_tokens=128).assemble(toy_record)])
    output = model(batch)
    for level, logits in enumerate(output.level_logits):
        start = model.semantic_offsets[level] + 1
        end = model.semantic_offsets[level + 1] + 1
        weight = model.semantic_embedding.weight[start:end] if tied else model.level_heads[level].weight
        torch.testing.assert_close(logits, output.hidden_states @ weight.T)
    assert output.loss is not None
    output.loss.backward()
    # SID 256 never occurs in the input. Only a tied classifier can update its embedding.
    gradient = model.semantic_embedding.weight.grad
    assert gradient[0].count_nonzero() == 0
    assert bool(gradient[256].count_nonzero()) == tied


def test_beam_generation_uses_level_specific_global_ranges(toy_record):
    assembler = SequenceAssembler(maximum_tokens=128)
    prefix = pack_sequences([assembler.assemble(toy_record, mode="test_prefix")])
    model = LGMPSModel(_settings()).eval()
    result = generate_semantic_ids(model, prefix, beam_size=3, return_sequences=3)
    assert len(result.semantic_ids) == 3
    assert result.scores == sorted(result.scores, reverse=True)
    for level_1, level_2, level_3 in result.semantic_ids:
        assert 1 <= level_1 <= 256
        assert 257 <= level_2 <= 512
        assert 513 <= level_3 <= 768


def test_batch_generation_fallback_matches_individual_generation(toy_record):
    assembler = SequenceAssembler(maximum_tokens=128)
    prefix = pack_sequences([assembler.assemble(toy_record, mode="test_prefix")])
    model = LGMPSModel(_settings()).eval()
    individual = generate_semantic_ids(model, prefix, beam_size=3, return_sequences=3)
    batched = generate_semantic_ids_batch(
        model,
        [prefix, prefix],
        beam_size=3,
        return_sequences=3,
    )

    assert batched == [individual, individual]


def test_session_causal_cached_generation_matches_full_recompute(toy_record):
    assembler = SequenceAssembler(maximum_tokens=128)
    prefix = pack_sequences([assembler.assemble(toy_record, mode="test_prefix")])
    settings = ModelSettings(
        attention_pattern="session_causal_transformer",
        query_vocabulary=100,
        width=32,
        layers=1,
        head_width=8,
        attention_heads=2,
        feedforward_width=48,
        use_triton_session_attention=False,
        use_flex_training_attention=False,
    )
    torch.manual_seed(29)
    model = LGMPSModel(settings).eval()
    cached = generate_semantic_ids(
        model,
        prefix,
        beam_size=3,
        return_sequences=3,
        use_cache=True,
    )
    recomputed = generate_semantic_ids(
        model,
        prefix,
        beam_size=3,
        return_sequences=3,
        use_cache=False,
    )
    assert cached.semantic_ids == recomputed.semantic_ids
    assert cached.scores == pytest.approx(recomputed.scores, abs=2e-5, rel=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FLA DeltaNet cache requires CUDA")
def test_deltanet_cached_generation_matches_full_recompute(toy_record):
    assembler = SequenceAssembler(maximum_tokens=128)
    prefix = pack_sequences([assembler.assemble(toy_record, mode="test_prefix")]).to("cuda")
    settings = ModelSettings(
        attention_pattern="para_chan_film_deltanet",
        query_vocabulary=100,
        width=32,
        layers=1,
        head_width=8,
        attention_heads=4,
        local_heads=2,
        history_heads=2,
        history_value_expansion=2,
        feedforward_width=48,
        history_backend="fla",
        delta_output_gate=False,
        use_triton_session_attention=True,
    )
    torch.manual_seed(31)
    model = LGMPSModel(settings).cuda().eval()
    cached = generate_semantic_ids(
        model,
        prefix,
        beam_size=3,
        return_sequences=3,
        use_cache=True,
    )
    recomputed = generate_semantic_ids(
        model,
        prefix,
        beam_size=3,
        return_sequences=3,
        use_cache=False,
    )
    assert cached.semantic_ids == recomputed.semantic_ids
    assert cached.scores == pytest.approx(recomputed.scores, abs=3e-3, rel=3e-3)
