from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from .common.artifacts import build_residual_quantizer, embed_items, write_item_texts
from .common.io import LFS_SIGNATURE
from .common.records import CANONICAL_FIELDS, CanonicalJsonlWriter, Document, Session, validate_record
from .common.semantic_ids import SID_FIELDS, read_semantic_id_map
from .common.tokenization import HashSmokeTextEncoder, SimpleBPE
from .jdsearch.adapter import BEHAVIOR_FIELDS
from .jdsearch.adapter import convert_sessions as convert_jdsearch
from .kuaisar.adapter import convert_sessions as convert_kuaisar
from .qilin.adapter import convert_sessions as convert_qilin
from .qilin.adapter import stable_user_id


def _write_csv(path: Path, fields: tuple[str, ...] | list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_and_validate(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    assert [record["record_id"] for record in records] == list(range(len(records)))
    for record in records:
        assert tuple(record) == CANONICAL_FIELDS
        assert record["item_identifiers"] is not None
        assert all(record["item_identifiers"])
        validate_record(record)
    return records


def _sid_fixture(path: Path) -> None:
    identifiers = ("k1", "k2", "k3", "j1", "j2", "j3", "q1", "q2", "q3")
    _write_csv(
        path,
        list(SID_FIELDS),
        [
            {
                "item_identifier": identifier,
                "semantic_id_1": index % 4,
                "semantic_id_2": (index + 1) % 4,
                "semantic_id_3": (index + 2) % 4,
            }
            for index, identifier in enumerate(identifiers)
        ],
    )


def _check_kuaisar(root: Path, sid_file: Path) -> int:
    source = root / "kuaisar.csv"
    fields = [
        "user_id",
        "search_session_id",
        "search_session_timestamp",
        "keyword",
        "search_source",
        "item_id",
        "click_cnt",
        "item_type",
    ]
    _write_csv(
        source,
        fields,
        [
            {
                "user_id": 7,
                "search_session_id": "a",
                "search_session_timestamp": 10,
                "keyword": "[1,2]",
                "search_source": "USER_INPUT",
                "item_id": "k1",
                "click_cnt": 1,
                "item_type": "VIDEO",
            },
            {
                "user_id": 7,
                "search_session_id": "b",
                "search_session_timestamp": 20,
                "keyword": "[3]",
                "search_source": "RELATED_FEED_QUERY",
                "item_id": "k2",
                "click_cnt": 0,
                "item_type": "IMAGE_ATLAS",
            },
            {
                "user_id": 8,
                "search_session_id": "c",
                "search_session_timestamp": 30,
                "keyword": "[4]",
                "search_source": "USER_INPUT",
                "item_id": "k3",
                "click_cnt": 1,
                "item_type": "VIDEO",
            },
        ],
    )
    output = root / "kuaisar.jsonl"
    summary = convert_kuaisar(source, sid_file, output, minimum_sessions=2)
    records = _read_and_validate(output)
    assert len(records) == 1 and records[0]["user_id"] == 7
    assert summary["filtered_short_chunks"] == 1
    return len(records)


def _check_jdsearch(root: Path, sid_file: Path) -> int:
    source = root / "jdsearch.tsv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BEHAVIOR_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "query": "[7,8]",
                "candidate_wid_list": "j3",
                "candidate_label_list": "1",
                "history_qry_list": "[1]_[2]",
                "history_wid_list": "j1_j2",
                "history_type_list": "CLICK_CART",
                "history_time_list": "0_1_2",
            }
        )
    output = root / "jdsearch.jsonl"
    convert_jdsearch(source, sid_file, output, minimum_sessions=2)
    records = _read_and_validate(output)
    assert len(records) == 1 and records[0]["user_id"] == 1
    assert records[0]["chunk_index"] == 0
    return len(records)


def _qilin_row(
    user: str, query: str, identifier: str, timestamp_name: str, timestamp: int, detail_name: str
) -> dict[str, object]:
    return {
        "user_id": user,
        "query": query,
        timestamp_name: timestamp,
        detail_name: json.dumps([{"note_idx": identifier, "click": 1, timestamp_name: timestamp}]),
    }


def _check_qilin(root: Path, sid_file: Path) -> int:
    search = root / "search.csv"
    dqa = root / "dqa.csv"
    recommendation = root / "recommendation.csv"
    _write_csv(
        search,
        ["user_id", "query", "search_timestamp", "search_result_details_with_idx"],
        [_qilin_row("person-a", "搜索", "q1", "search_timestamp", 10, "search_result_details_with_idx")],
    )
    _write_csv(
        dqa,
        ["user_id", "query", "search_timestamp", "search_result_details_with_idx"],
        [_qilin_row("person-a", "问答", "q2", "search_timestamp", 20, "search_result_details_with_idx")],
    )
    _write_csv(
        recommendation,
        ["user_id", "query", "request_timestamp", "rec_result_details_with_idx"],
        [_qilin_row("person-a", "", "q3", "request_timestamp", 30, "rec_result_details_with_idx")],
    )
    output = root / "qilin.jsonl"
    user_map = root / "qilin-users.csv"
    convert_qilin(
        search_files=[search],
        dqa_files=[dqa],
        recommendation_files=[recommendation],
        sid_file=sid_file,
        output_file=output,
        query_encoder=HashSmokeTextEncoder(),
        scope="all_requests",
        sid_modality="text",
        user_map_file=user_map,
        minimum_sessions=2,
    )
    records = _read_and_validate(output)
    assert len(records) == 1 and records[0]["user_id"] == stable_user_id("person-a")
    assert records[0]["session_attributes"] == [1, 2, 3]
    return len(records)


def _check_oversized_request(root: Path) -> int:
    documents = tuple(Document(str(index), (1, 257, 513), 0, 1) for index in range(1022))
    session = Session((1, 2), 1, documents)
    output = root / "oversized.jsonl"
    with CanonicalJsonlWriter(output, minimum_sessions=1) as writer:
        writer.write_user(user_id=1, dataset="fixture", sessions=[session])
    records = _read_and_validate(output)
    assert len(records) == 2
    assert records[0]["query_token_ids"][:2] == records[1]["query_token_ids"][:2] == [1, 2]
    assert [len(record["semantic_ids"]) for record in records] == [1021, 1]
    return len(records)


def _check_lfs_detection(root: Path) -> None:
    pointer = root / "pointer.csv"
    pointer.write_bytes(LFS_SIGNATURE + b"\noid sha256:fixture\nsize 1\n")
    try:
        read_semantic_id_map(pointer)
    except RuntimeError as error:
        assert "Git-LFS pointer" in str(error)
    else:
        raise AssertionError("Git-LFS pointer was not rejected")


def _check_artifact_stages(root: Path) -> tuple[int, int]:
    bpe = SimpleBPE()
    seen = bpe.train(
        ([1, 2, 1], [1, 2, 3], [1, 2, 1]),
        vocabulary_size=8,
        minimum_frequency=1,
    )
    bpe_path = root / "integer-bpe.json"
    bpe.save(bpe_path)
    restored = SimpleBPE.load(bpe_path)
    assert restored.encode_tokens([1, 2, 1]) == bpe.encode_tokens([1, 2, 1])

    item_text_file = root / "items.csv"
    write_item_texts(
        item_text_file,
        [(f"artifact-{index}", f"fixture item {index}", "") for index in range(4)],
    )
    embeddings = root / "embeddings.npy"
    identifiers = root / "identifiers.npy"
    item_count, dimension = embed_items(
        item_text_file,
        embeddings,
        identifiers,
        backend="hash-smoke",
        model_name="",
        device="cpu",
        batch_size=2,
        hash_dimension=8,
        modality="text",
        image_root=None,
    )
    assert (item_count, dimension) == (4, 8)
    sid_file = root / "artifact-semantic-ids.csv"
    build_residual_quantizer(
        embeddings,
        identifiers,
        sid_file,
        num_centroids=2,
        iterations=3,
    )
    assert len(read_semantic_id_map(sid_file, num_centroids=2)) == 4
    return seen, item_count


def run_self_check() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="lgmps-data-proc-") as directory:
        root = Path(directory)
        sid_file = root / "semantic-ids.csv"
        _sid_fixture(sid_file)
        results = {
            "kuaisar_records": _check_kuaisar(root, sid_file),
            "jdsearch_records": _check_jdsearch(root, sid_file),
            "qilin_records": _check_qilin(root, sid_file),
            "oversized_records": _check_oversized_request(root),
        }
        _check_lfs_detection(root)
        bpe_sequences, artifact_items = _check_artifact_stages(root)
        results["bpe_sequences"] = bpe_sequences
        results["artifact_items"] = artifact_items
    return {"status": "ok", **results}
