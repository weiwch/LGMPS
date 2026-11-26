from __future__ import annotations

import json
import mmap
import os
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from .schema import SessionSequenceRecord

try:
    import orjson
except ImportError:  # pragma: no cover - standard-library fallback
    orjson = None


def _default_index_path(data_path: Path) -> Path:
    return data_path.with_suffix(data_path.suffix + ".offsets.npy")


def _metadata_path(index_path: Path) -> Path:
    return index_path.with_suffix(index_path.suffix + ".meta.json")


def build_offset_index(data_file: str | Path, index_file: str | Path | None = None) -> Path:
    data_path = Path(data_file).expanduser().resolve()
    index_path = Path(index_file).expanduser().resolve() if index_file else _default_index_path(data_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    offsets: list[int] = []
    cursor = 0
    with data_path.open("rb") as handle:
        for line in handle:
            if line.strip():
                offsets.append(cursor)
            cursor += len(line)
    np.save(index_path, np.asarray(offsets, dtype=np.uint64), allow_pickle=False)
    stat = data_path.stat()
    metadata = {
        "data_file": os.path.relpath(data_path, start=index_path.parent),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "records": len(offsets),
    }
    with _metadata_path(index_path).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return index_path


def _index_is_current(data_path: Path, index_path: Path) -> bool:
    meta_path = _metadata_path(index_path)
    if not index_path.exists() or not meta_path.exists():
        return False
    try:
        with meta_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        stat = data_path.stat()
        return metadata.get("size_bytes") == stat.st_size and metadata.get("mtime_ns") == stat.st_mtime_ns
    except (OSError, ValueError, TypeError):
        return False


class IndexedJsonlDataset(Dataset[SessionSequenceRecord]):
    """Random-access JSONL dataset backed by a persistent byte-offset index and mmap."""

    def __init__(
        self,
        data_file: str | Path,
        index_file: str | Path | None = None,
        *,
        auto_index: bool = True,
        validate_records: bool = True,
        dataset_name: str = "",
    ) -> None:
        self.data_path = Path(data_file).expanduser().resolve()
        self.index_path = (
            Path(index_file).expanduser().resolve() if index_file else _default_index_path(self.data_path)
        )
        if not _index_is_current(self.data_path, self.index_path):
            if not auto_index:
                raise FileNotFoundError(f"Missing or stale offset index: {self.index_path}")
            build_offset_index(self.data_path, self.index_path)
        self.offsets = np.load(self.index_path, mmap_mode="r", allow_pickle=False)
        self.validate_records = validate_records
        self.dataset_name = dataset_name
        self._handle = None
        self._mapping = None

    def __len__(self) -> int:
        return int(self.offsets.shape[0])

    def _ensure_open(self) -> mmap.mmap:
        if self._mapping is None:
            self._handle = self.data_path.open("rb")
            self._mapping = mmap.mmap(self._handle.fileno(), length=0, access=mmap.ACCESS_READ)
        return self._mapping

    def __getitem__(self, index: int) -> SessionSequenceRecord:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        mapping = self._ensure_open()
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1]) if index + 1 < len(self) else mapping.size()
        raw = mapping[start:end].strip()
        payload: dict[str, Any] = orjson.loads(raw) if orjson is not None else json.loads(raw)
        return SessionSequenceRecord.from_mapping(
            payload,
            record_id=index,
            dataset=self.dataset_name or None,
            validate=self.validate_records,
        )

    def close(self) -> None:
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except (AttributeError, BufferError, OSError):
            return

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_mapping"] = None
        return state
