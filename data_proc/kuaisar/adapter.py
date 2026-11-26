from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..common.artifacts import write_item_texts
from ..common.io import as_int, first_nonempty, is_positive, iter_table_rows
from ..common.records import CanonicalJsonlWriter, Document, Session
from ..common.semantic_ids import read_semantic_id_map, semantic_ids_for
from ..common.tokenization import IntegerQueryEncoder, SimpleBPE, parse_integer_tokens

DATASET_NAME = "KuaiSAR_v2"
SOURCE_CODES = {
    "USER_INPUT": 1,
    "RELATED_FEED_QUERY": 2,
    "HOMEPAGE_GUESS": 3,
    "RECOMMEND": 4,
    "OTHER": 5,
    "IMAGE_ATLAS": 6,
    "VIDEO": 7,
}
ITEM_CATEGORY_CODES = {"IMAGE_ATLAS": 1, "VIDEO": 2}


def _sortable_time(value: Any, fallback: int) -> tuple[int, float | str]:
    try:
        return 0, float(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        return (1, text) if text else (0, float(fallback))


def _timestamp(
    row: dict[str, Any], fallback: int
) -> tuple[tuple[int, float | str], tuple[int, float | str], int]:
    session_time = first_nonempty(row, ("search_session_timestamp", "timestamp"), fallback)
    interaction_time = first_nonempty(row, ("time",), session_time)
    return _sortable_time(session_time, fallback), _sortable_time(interaction_time, fallback), fallback


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
) -> dict[str, int]:
    encoder = query_encoder or IntegerQueryEncoder()
    sid_map = read_semantic_id_map(sid_file)
    grouped: dict[
        int,
        dict[
            str,
            list[
                tuple[
                    tuple[tuple[int, float | str], tuple[int, float | str], int],
                    int,
                    dict[str, Any],
                ]
            ],
        ],
    ] = defaultdict(lambda: defaultdict(list))
    for row_index, row in enumerate(iter_table_rows(source_file)):
        if limit and row_index >= limit:
            break
        raw_user = first_nonempty(row, ("user_id",), None)
        if raw_user is None:
            raise ValueError(f"KuaiSAR row {row_index + 2} is missing user_id")
        user_id = as_int(raw_user, default=-1)
        if user_id < 0:
            raise ValueError(f"KuaiSAR row {row_index + 2} has an invalid user_id")
        session_identifier = str(first_nonempty(row, ("search_session_id",), f"row-{row_index}")).strip()
        grouped[user_id][session_identifier].append((_timestamp(row, row_index), row_index, row))

    with CanonicalJsonlWriter(
        output_file,
        maximum_tokens=maximum_tokens,
        minimum_sessions=minimum_sessions,
        overwrite=overwrite,
    ) as writer:
        for user_id in sorted(grouped):
            user_sessions: list[
                tuple[tuple[tuple[int, float | str], tuple[int, float | str], int], Session]
            ] = []
            for rows in grouped[user_id].values():
                rows.sort(key=lambda entry: (entry[0], entry[1]))
                first_row = rows[0][2]
                query_tokens = tuple(encoder.encode(first_row.get("keyword", ""), maximum_tokens=10))
                source = str(
                    first_nonempty(first_row, ("search_source", "search_session_source"), "OTHER")
                ).upper()
                documents: list[Document] = []
                for _, _, row in rows:
                    identifier = str(first_nonempty(row, ("item_id",), "")).strip()
                    if not identifier:
                        raise ValueError("KuaiSAR contains an interaction with no item_id")
                    item_type = str(row.get("item_type", "")).upper()
                    documents.append(
                        Document(
                            item_identifier=identifier,
                            semantic_ids=semantic_ids_for(identifier, sid_map),
                            interaction_label=1 if is_positive(row.get("click_cnt")) else 0,
                            item_category=ITEM_CATEGORY_CODES.get(item_type, 0),
                        )
                    )
                user_sessions.append(
                    (
                        rows[0][0],
                        Session(
                            query_token_ids=query_tokens,
                            session_attribute=SOURCE_CODES.get(source, 0),
                            documents=tuple(documents),
                        ),
                    )
                )
            user_sessions.sort(key=lambda entry: entry[0])
            writer.write_user(
                user_id=user_id,
                dataset=DATASET_NAME,
                sessions=[session for _, session in user_sessions],
            )
    return writer.summary_dict()


def _clean(value: Any) -> str:
    text = str(value).replace("\n", " ").strip()
    return "" if text.lower() in {"nan", "none"} else text


def extract_item_texts(
    source_file: str | Path,
    output_file: str | Path,
    *,
    bpe: SimpleBPE | None = None,
    limit: int = 0,
) -> int:
    rows: dict[str, tuple[str, str, str]] = {}
    for row_index, row in enumerate(iter_table_rows(source_file)):
        if limit and row_index >= limit:
            break
        identifier = str(first_nonempty(row, ("item_id",), "")).strip()
        if not identifier or identifier in rows:
            continue
        caption_tokens = parse_integer_tokens(row.get("caption", ""))
        if bpe is not None:
            caption_tokens = bpe.encode_tokens(caption_tokens)
        categories = [
            _clean(first_nonempty(row, (f"{level}_level_category_name_en", f"{level}_level_category_name")))
            for level in ("first", "second", "third", "fourth")
        ]
        parts = [
            f"caption tokens: {' '.join(str(token) for token in caption_tokens)}",
            f"item type: {_clean(row.get('item_type', ''))}",
            f"upload type: {_clean(row.get('upload_type', ''))}",
            f"categories: {', '.join(value for value in categories if value)}",
        ]
        rows[identifier] = (identifier, ". ".join(part for part in parts if part), "")
    return write_item_texts(output_file, [rows[key] for key in sorted(rows)])


def iter_bpe_sequences(
    behavior_file: str | Path,
    item_file: str | Path | None = None,
) -> Iterator[list[int]]:
    for row in iter_table_rows(behavior_file):
        tokens = parse_integer_tokens(row.get("keyword", ""))
        if tokens:
            yield tokens
    if item_file is not None:
        for row in iter_table_rows(item_file):
            tokens = parse_integer_tokens(row.get("caption", ""))
            if tokens:
                yield tokens
