from __future__ import annotations

import json

from lgmps.data import IndexedJsonlDataset, SequenceAssembler, convert_legacy_jsonl, pack_sequences


def test_chronological_assembly_and_packing(toy_record):
    toy_record.validate()
    assembler = SequenceAssembler(maximum_tokens=128)
    training = assembler.assemble(toy_record, mode="training")
    prefix = assembler.assemble(toy_record, mode="test_prefix")
    assert training.length == 23
    assert prefix.length == 28
    assert int((training.train_labels != -100).sum()) == 3
    assert int((training.validation_labels != -100).sum()) == 3
    assert prefix.target_semantic_ids == [(5, 261, 517)]
    packed = pack_sequences([training, training])
    assert packed.sequence_offsets.tolist() == [0, 23, 46]
    assert packed.session_offsets.tolist() == [0, 12, 23, 35, 46]
    assert packed.record_ids.tolist() == [0, 0]


def test_unversioned_records_can_be_converted_and_indexed(tmp_path, toy_record):
    payload = toy_record.to_mapping()
    for field in ("schema_version", "record_id", "dataset", "chunk_index", "item_identifiers"):
        del payload[field]
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    output = tmp_path / "canonical.jsonl"
    index = tmp_path / "offsets.npy"
    summary = convert_legacy_jsonl(source, output, index_file=index)
    assert summary.written_records == 1
    record = IndexedJsonlDataset(output, index)[0]
    assert record.record_id == 0
    assert record.user_id == 7
    assert record.chunk_index == 0
    assert record.session_count == 3
