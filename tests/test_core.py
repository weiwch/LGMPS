from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from lgmps.configuration import ModelSettings, load_experiment, repository_root
from lgmps.data import IndexedJsonlDataset, SequenceAssembler, build_offset_index, pack_sequences
from lgmps.data.schema import SessionSequenceRecord
from lgmps.nn import LGMPSModel
from lgmps.nn.attention import SegmentedSelfAttention


def test_canonical_record_round_trip_and_legacy_tolerance(tmp_path, toy_record) -> None:
    data_file = tmp_path / "records.jsonl"
    data_file.write_text(json.dumps(toy_record.to_mapping()) + "\n", encoding="utf-8")
    index_file = build_offset_index(data_file)
    dataset = IndexedJsonlDataset(data_file, index_file)
    loaded = dataset[0]
    assert loaded.to_mapping() == toy_record.to_mapping()

    unexpected = toy_record.to_mapping() | {"unexpected_field": []}
    with pytest.raises(ValueError, match="Unknown sequence fields"):
        SessionSequenceRecord.from_mapping(unexpected)

    incomplete = toy_record.to_mapping()
    incomplete.pop("item_identifiers")
    with pytest.raises(ValueError, match="Missing sequence fields"):
        SessionSequenceRecord.from_mapping(incomplete)

    legacy = toy_record.to_mapping()
    for field in ("schema_version", "dataset", "chunk_index", "item_identifiers"):
        legacy.pop(field)
    converted = SessionSequenceRecord.from_mapping(legacy, dataset="KuaiSAR_v2")
    assert converted.record_id == toy_record.record_id
    assert converted.item_identifiers is None


def test_mix_3_1_uses_aligned_eight_layer_pattern_with_sequential_rope(toy_record) -> None:
    settings = ModelSettings(
        attention_pattern="mix_3_1",
        query_vocabulary=100,
        width=32,
        layers=8,
        head_width=8,
        attention_heads=12,
        history_heads=6,
        history_value_expansion=2,
        feedforward_width=48,
        use_rope=True,
        rope_position_mode="sequential",
        history_backend="reference",
        use_triton_session_attention=False,
        use_flex_training_attention=False,
    )
    model = LGMPSModel(settings)
    assert [block.pattern for block in model.blocks] == [
        "gated_deltanet",
        "gated_deltanet",
        "gated_deltanet",
        "full_transformer",
        "gated_deltanet",
        "gated_deltanet",
        "gated_deltanet",
        "full_transformer",
    ]
    assert all(
        block.attention is None
        or (block.attention.use_rope and block.attention.rope_position_mode == "sequential")
        for block in model.blocks
    )
    sequence = SequenceAssembler(maximum_tokens=128).assemble(toy_record)
    output = model(pack_sequences([sequence]))
    assert output.loss is not None
    output.loss.backward()
    assert torch.isfinite(model.query_embedding.weight.grad).all()


def test_full_causal_attention_cache_matches_dense_forward() -> None:
    torch.manual_seed(7)
    attention = SegmentedSelfAttention(
        16,
        2,
        8,
        pattern="causal",
        use_rope=True,
        rope_position_mode="sequential",
        use_triton=False,
    ).eval()
    hidden = torch.randn(5, 16)
    positions = torch.zeros(5, dtype=torch.long)
    item_ids = torch.tensor([0, 0, 1, 1, 1])
    session_ids = torch.ones(5, dtype=torch.long)
    dense = attention(
        hidden,
        rope_positions=positions,
        sequence_offsets=torch.tensor([0, 5], dtype=torch.int32),
        session_offsets=torch.tensor([0, 5], dtype=torch.int32),
        session_ids=session_ids,
        item_ids=item_ids,
    )[-1]
    cache = attention.prefill_cache(
        hidden[:4],
        rope_positions=positions[:4],
        item_ids=item_ids[:4],
        session_ids=session_ids[:4],
        sequence_offsets=torch.tensor([0, 4], dtype=torch.int32),
        current_session_starts=[0],
    )
    cached, _ = attention.decode_cached(
        hidden[4:],
        rope_positions=positions[4:],
        cache=cache,
    )
    assert torch.allclose(cached[0], dense, atol=1e-5, rtol=1e-5)


def test_repository_root() -> None:
    root = Path(__file__).resolve().parents[1]
    assert repository_root() == root


def test_experiment_configs_select_expected_attention_patterns() -> None:
    root = repository_root()
    primary = load_experiment(root / "configs/para_chan_film.json", resolve_paths=False)
    ablation = load_experiment(root / "configs/deltanet_matched.json", resolve_paths=False)
    assert primary.model.attention_pattern == "para_chan_film"
    assert ablation.model.attention_pattern == "para_chan_film_deltanet"


def test_experiment_config_binds_dataset_paths() -> None:
    root = repository_root()
    bound = load_experiment(
        root / "configs/lgmps_dense.json",
        dataset="jdsearch",
        resolve_paths=False,
    )
    assert bound.dataset == "jdsearch"
    assert bound.data.train_file == "data/processed/jdsearch/session_sequences.jsonl"
    assert bound.training.output_dir == "outputs/jdsearch/lgmps_dense"
