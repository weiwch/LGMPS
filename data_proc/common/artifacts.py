from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .io import reject_lfs_pointer
from .semantic_ids import write_semantic_ids
from .tokenization import stable_text_embedding

ITEM_TEXT_FIELDS = ("item_identifier", "text", "image_path")


def read_item_texts(path: str | Path, *, limit: int = 0) -> tuple[list[str], list[str], list[str]]:
    resolved = reject_lfs_pointer(path)
    identifiers: list[str] = []
    texts: list[str] = []
    image_paths: list[str] = []
    with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ITEM_TEXT_FIELDS:
            raise ValueError(f"{resolved} must have exactly this header: {','.join(ITEM_TEXT_FIELDS)}")
        seen: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            identifier = str(row["item_identifier"]).strip()
            if not identifier or identifier in seen:
                raise ValueError(f"{resolved}:{line_number} has an invalid or duplicate item_identifier")
            seen.add(identifier)
            identifiers.append(identifier)
            texts.append(str(row["text"]))
            image_paths.append(str(row["image_path"]).strip())
            if limit and len(identifiers) >= limit:
                break
    if not identifiers:
        raise ValueError(f"{resolved} contains no item text rows")
    return identifiers, texts, image_paths


def write_item_texts(path: str | Path, rows: list[tuple[str, str, str]]) -> int:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ITEM_TEXT_FIELDS)
        writer.writeheader()
        for identifier, text, image_path in rows:
            identifier = str(identifier).strip()
            if not identifier or identifier in seen:
                raise ValueError(f"invalid or duplicate item_identifier={identifier!r}")
            seen.add(identifier)
            writer.writerow({"item_identifier": identifier, "text": str(text), "image_path": str(image_path)})
    return len(rows)


def _load_gme(model_name: str, device: str) -> Any:
    try:
        import torch
        from transformers import AutoModel
    except ImportError as error:
        raise ImportError("the gme embedding backend requires torch and transformers") from error
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("the explicitly selected gme backend requires an available CUDA device")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    return model.eval()


def embed_items(
    input_file: str | Path,
    embeddings_output: str | Path,
    identifiers_output: str | Path,
    *,
    backend: str,
    model_name: str,
    device: str,
    batch_size: int,
    hash_dimension: int,
    modality: str,
    image_root: str | Path | None,
    limit: int = 0,
) -> tuple[int, int]:
    import numpy as np

    identifiers, texts, image_paths = read_item_texts(input_file, limit=limit)
    if backend == "hash-smoke":
        embeddings = np.asarray(
            [stable_text_embedding(text, dimension=hash_dimension) for text in texts], dtype=np.float32
        )
    elif backend == "gme":
        import torch

        model = _load_gme(model_name, device)
        chunks: list[Any] = []
        image_base = Path(image_root).expanduser().resolve() if image_root else None
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                if modality == "text":
                    output = model.get_text_embeddings(texts=batch_texts, is_query=False)
                else:
                    try:
                        from PIL import Image
                    except ImportError as error:
                        raise ImportError("with_image embeddings require Pillow") from error
                    batch_images = []
                    for raw_path in image_paths[start : start + batch_size]:
                        if not raw_path:
                            raise ValueError("with_image modality requires image_path for every item")
                        image_path = Path(raw_path)
                        if not image_path.is_absolute():
                            if image_base is None:
                                raise ValueError("relative image paths require --image-root")
                            image_path = image_base / image_path
                        reject_lfs_pointer(image_path)
                        batch_images.append(Image.open(image_path).convert("RGB"))
                    output = model.get_fused_embeddings(texts=batch_texts, images=batch_images)
                chunks.append(output.detach().float().cpu().numpy())
        embeddings = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    else:
        raise ValueError("backend must be explicitly selected as 'gme' or 'hash-smoke'")

    embedding_path = Path(embeddings_output).expanduser().resolve()
    identifier_path = Path(identifiers_output).expanduser().resolve()
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    identifier_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embedding_path, embeddings, allow_pickle=False)
    np.save(identifier_path, np.asarray(identifiers, dtype=str), allow_pickle=False)
    return int(embeddings.shape[0]), int(embeddings.shape[1])


def _assign_nearest(values: Any, centroids: Any, *, batch_size: int = 8192) -> Any:
    import numpy as np

    labels = np.empty(values.shape[0], dtype=np.int32)
    centroid_norms = (centroids * centroids).sum(axis=1)
    for start in range(0, values.shape[0], batch_size):
        batch = values[start : start + batch_size]
        distances = (
            (batch * batch).sum(axis=1, keepdims=True) + centroid_norms[None, :] - 2 * batch.dot(centroids.T)
        )
        labels[start : start + len(batch)] = distances.argmin(axis=1).astype(np.int32)
    return labels


def _kmeans(values: Any, *, clusters: int, iterations: int, seed: int) -> tuple[Any, Any]:
    import numpy as np

    generator = np.random.default_rng(seed)
    cluster_count = min(int(clusters), int(values.shape[0]))
    centroids = values[generator.choice(values.shape[0], size=cluster_count, replace=False)].copy()
    labels = np.full(values.shape[0], -1, dtype=np.int32)
    for _ in range(iterations):
        updated = _assign_nearest(values, centroids)
        if np.array_equal(labels, updated):
            labels = updated
            break
        labels = updated
        for cluster in range(cluster_count):
            selected = labels == cluster
            centroids[cluster] = (
                values[selected].mean(axis=0)
                if selected.any()
                else values[generator.integers(values.shape[0])]
            )
    return labels, centroids


def build_residual_quantizer(
    embeddings_file: str | Path,
    identifiers_file: str | Path,
    output_file: str | Path,
    *,
    num_centroids: int = 256,
    iterations: int = 20,
    seed: int = 2026,
    codebooks_output: str | Path | None = None,
) -> int:
    import numpy as np

    embedding_path = reject_lfs_pointer(embeddings_file)
    identifier_path = reject_lfs_pointer(identifiers_file)
    embeddings = np.load(embedding_path, allow_pickle=False).astype(np.float32, copy=False)
    identifiers = np.load(identifier_path, allow_pickle=False)
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty rank-two array")
    if identifiers.ndim != 1 or identifiers.shape[0] != embeddings.shape[0]:
        raise ValueError("identifier and embedding arrays are not aligned")

    residuals = embeddings.copy()
    codes = np.empty((embeddings.shape[0], 3), dtype=np.int32)
    codebooks: dict[str, Any] = {}
    for level in range(3):
        labels, centroids = _kmeans(
            residuals,
            clusters=num_centroids,
            iterations=iterations,
            seed=seed + level,
        )
        codes[:, level] = labels
        residuals -= centroids[labels]
        codebooks[f"level_{level + 1}"] = centroids
    write_semantic_ids(output_file, [str(value) for value in identifiers.tolist()], codes.tolist())
    if codebooks_output:
        codebook_path = Path(codebooks_output).expanduser().resolve()
        codebook_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(codebook_path, **codebooks)
    return int(embeddings.shape[0])
