from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..common.artifacts import write_item_texts
from ..common.io import first_nonempty, is_positive, iter_delimited_rows
from ..common.records import CanonicalJsonlWriter, Document, Session
from ..common.semantic_ids import read_semantic_id_map, semantic_ids_for
from ..common.tokenization import IntegerQueryEncoder, SimpleBPE, parse_integer_tokens

DATASET_NAME = "JDsearch"
BEHAVIOR_FIELDS = (
    "query",
    "candidate_wid_list",
    "candidate_label_list",
    "history_qry_list",
    "history_wid_list",
    "history_type_list",
    "history_time_list",
)
PRODUCT_FIELDS = (
    "wid",
    "name",
    "brand_id",
    "brand_name",
    "cate_id_1",
    "cate_name_1",
    "cate_id_2",
    "cate_name_2",
    "cate_id_3",
    "cate_name_3",
    "cate_id_4",
    "cate_name_4",
    "shop_id",
)
POSITIVE_HISTORY_TYPES = {"ORD", "CLICK", "CART", "FLW"}


def _list(value: Any) -> list[str]:
    text = str(value or "")
    return [] if not text else text.split("_")


def _build_sessions(
    row: dict[str, str],
    encoder: IntegerQueryEncoder,
    sid_map: dict[str, tuple[int, int, int]],
    *,
    skip_empty_query: bool,
) -> list[Session]:
    raw_sessions: list[tuple[float, str, list[tuple[str, int]], int]] = []
    history_queries = _list(row.get("history_qry_list"))
    history_items = _list(row.get("history_wid_list"))
    history_types = _list(row.get("history_type_list"))
    history_times = _list(row.get("history_time_list"))
    if not (len(history_queries) == len(history_items) == len(history_types)):
        raise ValueError("JDsearch history query, item, and type lists must have identical lengths")
    if len(history_times) != len(history_items) + 1:
        raise ValueError(
            "JDsearch history_time_list must contain one interval per history item plus current query"
        )
    try:
        intervals = [float(value) for value in history_times]
    except ValueError as error:
        raise ValueError("JDsearch history_time_list contains a non-numeric interval") from error
    if any(value < 0 for value in intervals):
        raise ValueError("JDsearch history_time_list cannot contain negative intervals")
    cumulative_times: list[float] = []
    elapsed = 0.0
    for interval in intervals:
        elapsed += interval
        cumulative_times.append(elapsed)

    current_query: str | None = None
    current_documents: list[tuple[str, int]] = []
    current_time = 0.0
    history_rows = sorted(
        zip(history_queries, history_items, history_types, cumulative_times[: len(history_items)]),
        key=lambda entry: entry[3],
    )
    for query, identifier, event_type, event_time in history_rows:
        event = 1 if str(event_type).upper() in POSITIVE_HISTORY_TYPES else 0
        if current_query is None:
            current_query = query
            current_documents.append((identifier, event))
            current_time = event_time
        elif query == current_query:
            current_documents.append((identifier, event))
            current_time = min(current_time, event_time)
        else:
            raw_sessions.append((current_time, current_query, current_documents, 1))
            current_query = query
            current_documents = [(identifier, event)]
            current_time = event_time
    if current_query is not None and current_documents:
        raw_sessions.append((current_time, current_query, current_documents, 1))

    candidate_items = _list(row.get("candidate_wid_list"))
    candidate_labels = _list(row.get("candidate_label_list"))
    if len(candidate_items) != len(candidate_labels):
        raise ValueError("JDsearch candidate item and label lists must have identical lengths")
    if candidate_items:
        raw_sessions.append(
            (
                cumulative_times[-1],
                row.get("query", ""),
                [
                    (identifier, 1 if is_positive(candidate_labels[index]) else 0)
                    for index, identifier in enumerate(candidate_items)
                ],
                2,
            )
        )

    sessions: list[Session] = []
    for _, query, raw_documents, origin_code in sorted(raw_sessions, key=lambda entry: entry[0]):
        query_tokens = tuple(encoder.encode(query, maximum_tokens=10))
        if skip_empty_query and not query_tokens:
            continue
        documents = tuple(
            Document(
                item_identifier=str(identifier),
                semantic_ids=semantic_ids_for(str(identifier), sid_map),
                interaction_label=event,
                item_category=1,
            )
            for identifier, event in raw_documents
            if str(identifier).strip()
        )
        if documents:
            sessions.append(
                Session(
                    query_token_ids=query_tokens,
                    session_attribute=origin_code,
                    documents=documents,
                )
            )
    return sessions


def convert_sessions(
    source_file: str | Path,
    sid_file: str | Path,
    output_file: str | Path,
    *,
    query_encoder: IntegerQueryEncoder | None = None,
    maximum_tokens: int = 4096,
    minimum_sessions: int = 2,
    overwrite: bool = False,
    limit: int = 0,
    skip_empty_query: bool = True,
) -> dict[str, int]:
    encoder = query_encoder or IntegerQueryEncoder()
    sid_map = read_semantic_id_map(sid_file)
    with CanonicalJsonlWriter(
        output_file,
        maximum_tokens=maximum_tokens,
        minimum_sessions=minimum_sessions,
        overwrite=overwrite,
    ) as writer:
        rows = iter_delimited_rows(source_file, delimiter="\t", fieldnames=BEHAVIOR_FIELDS)
        for row_index, row in enumerate(rows):
            if limit and row_index >= limit:
                break
            sessions = _build_sessions(row, encoder, sid_map, skip_empty_query=skip_empty_query)
            writer.write_user(user_id=row_index + 1, dataset=DATASET_NAME, sessions=sessions)
    return writer.summary_dict()


def extract_item_texts(
    source_file: str | Path,
    output_file: str | Path,
    *,
    bpe: SimpleBPE | None = None,
    limit: int = 0,
) -> int:
    output: dict[str, tuple[str, str, str]] = {}
    rows = iter_delimited_rows(source_file, delimiter="\t", fieldnames=PRODUCT_FIELDS)
    for row_index, row in enumerate(rows):
        if limit and row_index >= limit:
            break
        identifier = str(first_nonempty(row, ("wid",), "")).strip()
        if not identifier or identifier in output:
            continue

        def encoded(name: str, source_row: dict[str, str] = row) -> str:
            tokens = parse_integer_tokens(source_row.get(name, ""))
            if bpe is not None:
                tokens = bpe.encode_tokens(tokens)
            return " ".join(str(token) for token in tokens)

        categories = [encoded(f"cate_name_{level}") for level in range(1, 5)]
        text = ". ".join(
            (
                f"product title tokens: {encoded('name')}",
                f"brand tokens: {encoded('brand_name')}",
                f"category tokens: {', '.join(value for value in categories if value)}",
            )
        )
        output[identifier] = (identifier, text, "")
    return write_item_texts(output_file, [output[key] for key in sorted(output)])


def iter_bpe_sequences(
    behavior_file: str | Path,
    item_file: str | Path | None = None,
) -> Iterator[list[int]]:
    rows = iter_delimited_rows(behavior_file, delimiter="\t", fieldnames=BEHAVIOR_FIELDS)
    for row in rows:
        query = parse_integer_tokens(row.get("query", ""))
        if query:
            yield query
        for raw_query in _list(row.get("history_qry_list")):
            tokens = parse_integer_tokens(raw_query)
            if tokens:
                yield tokens
    if item_file is not None:
        rows = iter_delimited_rows(item_file, delimiter="\t", fieldnames=PRODUCT_FIELDS)
        for row in rows:
            for name in ("name", "brand_name", "cate_name_1", "cate_name_2", "cate_name_3", "cate_name_4"):
                tokens = parse_integer_tokens(row.get(name, ""))
                if tokens:
                    yield tokens
