from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from ..common.artifacts import write_item_texts
from ..common.io import (
    expand_input_paths,
    first_nonempty,
    is_positive,
    iter_table_rows,
    parse_jsonish,
)
from ..common.records import CanonicalJsonlWriter, Document, Session
from ..common.semantic_ids import read_semantic_id_map, semantic_ids_for
from ..common.tokenization import QueryEncoder

DATASET_NAME = "Qilin"
SESSION_KIND_CODES = {"search": 1, "dqa": 2, "recommendation": 3}
DETAIL_COLUMNS = {
    "search": "search_result_details_with_idx",
    "dqa": "search_result_details_with_idx",
    "recommendation": "rec_result_details_with_idx",
}


def stable_user_id(raw_identifier: object) -> int:
    text = str(raw_identifier).strip()
    if not text or text.lower() in {"nan", "none", "unknown"}:
        raise ValueError("Qilin row has no real user identifier")
    try:
        numeric_float = float(text)
        numeric = int(numeric_float)
        if numeric >= 0 and numeric_float == numeric:
            return numeric
    except (ValueError, OverflowError):
        pass
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8, person=b"lgmpsusr").digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def _details(raw_value: Any) -> list[dict[str, Any]]:
    parsed = parse_jsonish(raw_value)
    if parsed is None:
        return []
    if isinstance(parsed, Mapping):
        parsed = [parsed]
    if isinstance(parsed, str):
        raise TypeError("Qilin result details are not valid JSON/list data")
    try:
        values = list(parsed)
    except TypeError as error:
        raise ValueError("Qilin result details are not iterable") from error
    output: list[dict[str, Any]] = []
    for detail_value in values:
        if isinstance(detail_value, Mapping):
            output.append(dict(detail_value))
        elif hasattr(detail_value, "item"):
            converted = detail_value.item()
            if isinstance(converted, Mapping):
                output.append(dict(converted))
            else:
                raise TypeError("Qilin result detail is not an object")
        else:
            raise TypeError("Qilin result detail is not an object")
    return output


def _timestamp_key(value: Any, fallback: int) -> tuple[int, Any, int]:
    try:
        return 0, float(value), fallback
    except (TypeError, ValueError, OverflowError):
        pass
    text = str(value).strip()
    if text:
        try:
            return 0, datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp(), fallback
        except ValueError:
            return 1, text, fallback
    return 2, fallback, fallback


def _row_timestamp(kind: str, row: dict[str, Any], details: list[dict[str, Any]], row_index: int) -> Any:
    if kind == "recommendation":
        value = first_nonempty(details[0], ("request_timestamp",), None) if details else None
        if value is None:
            value = first_nonempty(row, ("request_timestamp",), None)
        if value is None:
            raise ValueError("Qilin recommendation rows require request_timestamp")
        return value
    value = first_nonempty(row, ("search_timestamp", "request_timestamp"), None)
    if value is not None:
        return value
    if details:
        return first_nonempty(details[0], ("search_timestamp",), row_index)
    return row_index


def _raw_sources(
    *,
    scope: str,
    search_files: Sequence[str | Path],
    dqa_files: Sequence[str | Path],
    recommendation_files: Sequence[str | Path],
) -> list[tuple[str, Path]]:
    requested = {
        "search": scope in {"all_requests", "search_dqa", "search"},
        "dqa": scope in {"all_requests", "search_dqa", "dqa"},
        "recommendation": scope in {"all_requests", "recommendation"},
    }
    supplied = {
        "search": search_files,
        "dqa": dqa_files,
        "recommendation": recommendation_files,
    }
    output: list[tuple[str, Path]] = []
    for kind in ("search", "dqa", "recommendation"):
        if not requested[kind]:
            continue
        if not supplied[kind]:
            raise ValueError(f"scope={scope!r} requires at least one {kind} input")
        output.extend((kind, path) for path in expand_input_paths(supplied[kind]))
    return output


def convert_sessions(
    *,
    search_files: Sequence[str | Path],
    dqa_files: Sequence[str | Path],
    recommendation_files: Sequence[str | Path],
    sid_file: str | Path,
    output_file: str | Path,
    query_encoder: QueryEncoder,
    scope: str = "all_requests",
    sid_modality: str = "text",
    user_map_file: str | Path | None = None,
    maximum_tokens: int = 4096,
    minimum_sessions: int = 2,
    overwrite: bool = False,
    limit: int = 0,
) -> dict[str, int]:
    if scope not in {"all_requests", "search_dqa", "search", "dqa", "recommendation"}:
        raise ValueError("unsupported Qilin scope")
    if sid_modality not in {"text", "with_image"}:
        raise ValueError("sid_modality must be 'text' or 'with_image'")
    mapping_path = Path(user_map_file).expanduser().resolve() if user_map_file is not None else None
    if mapping_path is not None and mapping_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {mapping_path}")
    sid_map = read_semantic_id_map(sid_file)
    grouped: dict[str, list[tuple[tuple[int, Any, int], int, Session]]] = defaultdict(list)
    total_rows = 0
    for source_rank, (kind, path) in enumerate(
        _raw_sources(
            scope=scope,
            search_files=search_files,
            dqa_files=dqa_files,
            recommendation_files=recommendation_files,
        )
    ):
        for row_index, row in enumerate(iter_table_rows(path)):
            if limit and total_rows >= limit:
                break
            total_rows += 1
            raw_user = first_nonempty(row, ("user_idx", "user_id"), None)
            if raw_user is None:
                raise ValueError(f"{path} contains a row without user identity")
            raw_user_text = str(raw_user).strip()
            details = _details(row.get(DETAIL_COLUMNS[kind]))
            if not details:
                raise ValueError(f"{path} contains a row with empty result details")
            documents: list[Document] = []
            for detail in details:
                identifier = str(first_nonempty(detail, ("note_idx", "note_id"), "")).strip()
                if not identifier:
                    raise ValueError(f"{path} contains a result without note identity")
                interaction = any(
                    is_positive(detail.get(name)) for name in ("click", "collect", "comment", "like", "share")
                )
                documents.append(
                    Document(
                        item_identifier=identifier,
                        semantic_ids=semantic_ids_for(identifier, sid_map),
                        interaction_label=1 if interaction else 0,
                        item_category=1,
                    )
                )
            query = "" if kind == "recommendation" else row.get("query", "")
            query_tokens = tuple(query_encoder.encode(query, maximum_tokens=10))
            if kind != "recommendation" and not query_tokens:
                continue
            timestamp = _row_timestamp(kind, row, details, row_index)
            grouped[raw_user_text].append(
                (
                    _timestamp_key(timestamp, row_index),
                    source_rank,
                    Session(
                        query_token_ids=query_tokens,
                        session_attribute=SESSION_KIND_CODES[kind],
                        documents=tuple(documents),
                    ),
                )
            )
        if limit and total_rows >= limit:
            break

    mappings: list[tuple[str, int]] = []
    reverse_ids: dict[int, str] = {}
    with CanonicalJsonlWriter(
        output_file,
        maximum_tokens=maximum_tokens,
        minimum_sessions=minimum_sessions,
        overwrite=overwrite,
    ) as writer:
        for raw_user in sorted(grouped):
            user_id = stable_user_id(raw_user)
            previous = reverse_ids.setdefault(user_id, raw_user)
            if previous != raw_user:
                raise RuntimeError(f"stable Qilin user ID collision between {previous!r} and {raw_user!r}")
            mappings.append((raw_user, user_id))
            sessions = sorted(grouped[raw_user], key=lambda entry: (entry[0], entry[1]))
            writer.write_user(
                user_id=user_id,
                dataset=DATASET_NAME,
                sessions=[session for _, _, session in sessions],
            )

    if mapping_path is not None:
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with mapping_path.open("w", encoding="utf-8", newline="") as handle:
            writer_csv = csv.writer(handle)
            writer_csv.writerow(("raw_user_identifier", "user_id"))
            writer_csv.writerows(mappings)
    summary = writer.summary_dict()
    summary["source_rows"] = total_rows
    return summary


def _first_image_path(value: Any) -> str:
    parsed = parse_jsonish(value)
    if isinstance(parsed, (list, tuple)):
        return str(parsed[0]) if parsed else ""
    return "" if parsed is None else str(parsed)


def extract_item_texts(
    source_files: Sequence[str | Path],
    output_file: str | Path,
    *,
    limit: int = 0,
) -> int:
    output: dict[str, tuple[str, str, str]] = {}
    for path in expand_input_paths(source_files):
        for row in iter_table_rows(path):
            identifier = str(first_nonempty(row, ("note_idx", "note_id"), "")).strip()
            if not identifier or identifier in output:
                continue
            title = str(row.get("note_title", "")).replace("\n", " ").strip()
            content = str(row.get("note_content", "")).replace("\n", " ").strip()
            taxonomies = ", ".join(
                str(row.get(name, "")).strip() for name in ("taxonomy1_id", "taxonomy2_id")
            )
            text = f"title: {title}. content: {content}. categories: {taxonomies}."
            output[identifier] = (identifier, text, _first_image_path(row.get("image_path")))
            if limit and len(output) >= limit:
                break
        if limit and len(output) >= limit:
            break
    return write_item_texts(output_file, [output[key] for key in sorted(output)])
