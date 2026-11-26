from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common.artifacts import build_residual_quantizer, embed_items
from .common.tokenization import (
    HashSmokeTextEncoder,
    HuggingFaceQueryEncoder,
    IntegerQueryEncoder,
    SimpleBPE,
)
from .jdsearch import adapter as jdsearch
from .kuaisar import adapter as kuaisar
from .qilin import adapter as qilin


def _add_write_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--maximum-tokens", type=int, default=4096)
    parser.add_argument("--minimum-sessions", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")


def _load_bpe(path: str) -> SimpleBPE | None:
    return SimpleBPE.load(path) if path else None


def _sessions_parser(subparsers: argparse._SubParsersAction) -> None:
    sessions = subparsers.add_parser("sessions", help="Build canonical session JSONL")
    datasets = sessions.add_subparsers(dest="dataset", required=True)

    kuai = datasets.add_parser("kuaisar", help="Convert KuaiSAR search interactions")
    kuai.add_argument("--source", required=True)
    kuai.add_argument("--sid-file", required=True)
    kuai.add_argument("--output", required=True)
    kuai.add_argument("--bpe-model", default="")
    _add_write_options(kuai)

    jd = datasets.add_parser("jdsearch", help="Convert JDsearch behavior TSV")
    jd.add_argument("--source", required=True)
    jd.add_argument("--sid-file", required=True)
    jd.add_argument("--output", required=True)
    jd.add_argument("--bpe-model", default="")
    jd.add_argument("--keep-empty-query", action="store_true")
    _add_write_options(jd)

    qi = datasets.add_parser("qilin", help="Convert Qilin request tables")
    qi.add_argument("--search", nargs="*", default=[])
    qi.add_argument("--dqa", nargs="*", default=[])
    qi.add_argument("--recommendation", nargs="*", default=[])
    qi.add_argument(
        "--scope",
        choices=("all_requests", "search_dqa", "search", "dqa", "recommendation"),
        default="all_requests",
    )
    qi.add_argument("--sid-modality", choices=("text", "with_image"), default="text")
    qi.add_argument("--sid-file", required=True)
    qi.add_argument("--output", required=True)
    qi.add_argument("--user-map", default="")
    qi.add_argument("--query-backend", choices=("huggingface", "hash-smoke"), required=True)
    qi.add_argument("--tokenizer", default="")
    qi.add_argument("--query-vocabulary", type=int, default=151_936)
    _add_write_options(qi)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m data_proc",
        description="Raw-data preparation for LGMPS",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _sessions_parser(subparsers)

    items = subparsers.add_parser("items", help="Extract canonical item text")
    items.add_argument("--dataset", choices=("kuaisar", "jdsearch", "qilin"), required=True)
    items.add_argument("--source", nargs="+", required=True)
    items.add_argument("--output", required=True)
    items.add_argument("--bpe-model", default="")
    items.add_argument("--limit", type=int, default=0)

    bpe = subparsers.add_parser("train-bpe", help="Train deterministic BPE for anonymized integer terms")
    bpe.add_argument("--dataset", choices=("kuaisar", "jdsearch"), required=True)
    bpe.add_argument("--source", required=True)
    bpe.add_argument("--item-source", default="")
    bpe.add_argument("--output", required=True)
    bpe.add_argument("--vocabulary-size", type=int, default=151_760)
    bpe.add_argument("--minimum-frequency", type=int, default=2)
    bpe.add_argument("--maximum-sequences", type=int, default=0)

    embed = subparsers.add_parser("embed", help="Embed canonical item text")
    embed.add_argument("--input", required=True)
    embed.add_argument("--embeddings-output", required=True)
    embed.add_argument("--identifiers-output", required=True)
    embed.add_argument("--backend", choices=("gme", "hash-smoke"), required=True)
    embed.add_argument("--model", default="Alibaba-NLP/gme-Qwen2-VL-2B-Instruct")
    embed.add_argument("--device", default="cuda:0")
    embed.add_argument("--batch-size", type=int, default=64)
    embed.add_argument("--hash-dimension", type=int, default=256)
    embed.add_argument("--modality", choices=("text", "with_image"), default="text")
    embed.add_argument("--image-root", default="")
    embed.add_argument("--limit", type=int, default=0)

    rq = subparsers.add_parser("rq", help="Train three-level residual quantization IDs")
    rq.add_argument("--embeddings", required=True)
    rq.add_argument("--identifiers", required=True)
    rq.add_argument("--output", required=True)
    rq.add_argument("--codebooks-output", default="")
    rq.add_argument("--num-centroids", type=int, default=256)
    rq.add_argument("--iterations", type=int, default=20)
    rq.add_argument("--seed", type=int, default=2026)

    subparsers.add_parser("self-check", help="Run all three adapters on synthetic fixtures")
    return parser


def _run_sessions(arguments: argparse.Namespace) -> dict[str, int]:
    common = {
        "maximum_tokens": arguments.maximum_tokens,
        "minimum_sessions": arguments.minimum_sessions,
        "overwrite": arguments.overwrite,
        "limit": arguments.limit,
    }
    if arguments.dataset == "kuaisar":
        return kuaisar.convert_sessions(
            arguments.source,
            arguments.sid_file,
            arguments.output,
            query_encoder=IntegerQueryEncoder(_load_bpe(arguments.bpe_model)),
            **common,
        )
    if arguments.dataset == "jdsearch":
        return jdsearch.convert_sessions(
            arguments.source,
            arguments.sid_file,
            arguments.output,
            query_encoder=IntegerQueryEncoder(_load_bpe(arguments.bpe_model)),
            skip_empty_query=not arguments.keep_empty_query,
            **common,
        )
    if arguments.query_backend == "huggingface":
        if not arguments.tokenizer:
            raise ValueError("--query-backend huggingface requires --tokenizer")
        encoder = HuggingFaceQueryEncoder(arguments.tokenizer)
    else:
        encoder = HashSmokeTextEncoder(arguments.query_vocabulary)
    return qilin.convert_sessions(
        search_files=arguments.search,
        dqa_files=arguments.dqa,
        recommendation_files=arguments.recommendation,
        sid_file=arguments.sid_file,
        output_file=arguments.output,
        query_encoder=encoder,
        scope=arguments.scope,
        sid_modality=arguments.sid_modality,
        user_map_file=arguments.user_map or None,
        **common,
    )


def run(arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.command == "sessions":
        return _run_sessions(arguments)
    if arguments.command == "items":
        bpe = _load_bpe(arguments.bpe_model)
        if arguments.dataset == "kuaisar":
            if len(arguments.source) != 1:
                raise ValueError("KuaiSAR item extraction accepts one item table")
            count = kuaisar.extract_item_texts(
                arguments.source[0], arguments.output, bpe=bpe, limit=arguments.limit
            )
        elif arguments.dataset == "jdsearch":
            if len(arguments.source) != 1:
                raise ValueError("JDsearch item extraction accepts one product TSV")
            count = jdsearch.extract_item_texts(
                arguments.source[0], arguments.output, bpe=bpe, limit=arguments.limit
            )
        else:
            if bpe is not None:
                raise ValueError("Qilin natural-language item text does not use the integer BPE stage")
            count = qilin.extract_item_texts(arguments.source, arguments.output, limit=arguments.limit)
        return {"items": count, "output": str(Path(arguments.output))}
    if arguments.command == "train-bpe":
        if arguments.dataset == "kuaisar":
            sequences = kuaisar.iter_bpe_sequences(arguments.source, arguments.item_source or None)
        else:
            sequences = jdsearch.iter_bpe_sequences(arguments.source, arguments.item_source or None)
        tokenizer = SimpleBPE()
        seen = tokenizer.train(
            sequences,
            vocabulary_size=arguments.vocabulary_size,
            minimum_frequency=arguments.minimum_frequency,
            maximum_sequences=arguments.maximum_sequences,
        )
        tokenizer.save(arguments.output)
        return {"sequences": seen, "vocabulary": len(tokenizer.vocabulary)}
    if arguments.command == "embed":
        rows, dimension = embed_items(
            arguments.input,
            arguments.embeddings_output,
            arguments.identifiers_output,
            backend=arguments.backend,
            model_name=arguments.model,
            device=arguments.device,
            batch_size=arguments.batch_size,
            hash_dimension=arguments.hash_dimension,
            modality=arguments.modality,
            image_root=arguments.image_root or None,
            limit=arguments.limit,
        )
        return {"items": rows, "embedding_dimension": dimension, "backend": arguments.backend}
    if arguments.command == "rq":
        count = build_residual_quantizer(
            arguments.embeddings,
            arguments.identifiers,
            arguments.output,
            num_centroids=arguments.num_centroids,
            iterations=arguments.iterations,
            seed=arguments.seed,
            codebooks_output=arguments.codebooks_output or None,
        )
        return {"items": count, "semantic_id_file": arguments.output}
    if arguments.command == "self-check":
        from .self_check import run_self_check

        return run_self_check()
    raise AssertionError(arguments.command)


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
