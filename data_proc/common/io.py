from __future__ import annotations

import ast
import csv
import glob
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

LFS_SIGNATURE = b"version https://git-lfs.github.com/spec/v1"


def reject_lfs_pointer(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with resolved.open("rb") as handle:
        prefix = handle.read(len(LFS_SIGNATURE))
    if prefix == LFS_SIGNATURE:
        raise RuntimeError(
            f"{resolved} is a Git-LFS pointer, not dataset content; run git lfs pull before preprocessing"
        )
    return resolved


def expand_input_paths(values: Sequence[str | Path]) -> list[Path]:
    resolved: set[Path] = set()
    for value in values:
        raw = str(value)
        matches = [Path(match) for match in glob.glob(raw, recursive=True)]
        if not matches:
            matches = [Path(raw)]
        for match in matches:
            if match.is_dir():
                for child in match.rglob("*"):
                    if child.suffix.lower() in {".csv", ".jsonl", ".parquet", ".tsv", ".txt"}:
                        resolved.add(child.resolve())
            elif match.exists():
                resolved.add(match.resolve())
            else:
                raise FileNotFoundError(match)
    paths = sorted(resolved)
    for path in paths:
        reject_lfs_pointer(path)
    return paths


def iter_delimited_rows(
    path: str | Path,
    *,
    delimiter: str = ",",
    fieldnames: Sequence[str] | None = None,
) -> Iterator[dict[str, str]]:
    resolved = reject_lfs_pointer(path)
    with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.reader(handle, delimiter=delimiter)
        try:
            first = next(rows)
        except StopIteration:
            return
        if fieldnames is None or set(fieldnames).issubset(set(first)):
            header = [str(value).strip() for value in first]
        else:
            header = list(fieldnames)
            yield {name: first[index] if index < len(first) else "" for index, name in enumerate(header)}
        for values in rows:
            if not values or not any(str(value).strip() for value in values):
                continue
            yield {name: values[index] if index < len(values) else "" for index, name in enumerate(header)}


def iter_table_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    resolved = reject_lfs_pointer(path)
    suffix = resolved.suffix.lower()
    if suffix == ".csv":
        yield from iter_delimited_rows(resolved)
        return
    if suffix in {".tsv", ".txt"}:
        yield from iter_delimited_rows(resolved, delimiter="\t")
        return
    if suffix == ".jsonl":
        with resolved.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise TypeError(f"{resolved}:{line_number} is not a JSON object")
                yield payload
        return
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as error:
            raise ImportError("reading Parquet inputs requires pandas and pyarrow") from error
        frame = pd.read_parquet(resolved)
        for payload in frame.to_dict(orient="records"):
            yield payload
        return
    raise ValueError(f"unsupported table format: {resolved}")


def parse_jsonish(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list, tuple)):
        return value
    if hasattr(value, "tolist") and not isinstance(value, str):
        return value.tolist()
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    return text


def as_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def is_positive(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError, OverflowError):
        return False


def first_nonempty(mapping: Mapping[str, Any], names: Iterable[str], default: Any = "") -> Any:
    for name in names:
        value = mapping.get(name)
        if value is not None and str(value).strip() and str(value).lower() != "nan":
            return value
    return default
