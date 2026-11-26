from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol

from .io import parse_jsonish, reject_lfs_pointer

INTEGER_TOKEN_SEPARATOR = "\x18"


def parse_integer_tokens(value: object) -> list[int]:
    parsed = parse_jsonish(value)
    if parsed is None:
        return []
    if isinstance(parsed, (list, tuple)):
        candidates = parsed
    else:
        text = str(parsed).strip()
        if not text or text == "-1":
            return []
        candidates = re.split(rf"(?:{re.escape(INTEGER_TOKEN_SEPARATOR)}|[\s,]+)", text)
    tokens: list[int] = []
    for candidate in candidates:
        try:
            token = int(candidate)
        except (TypeError, ValueError):
            continue
        if token >= 0:
            tokens.append(token)
    return tokens


class QueryEncoder(Protocol):
    def encode(self, value: object, *, maximum_tokens: int = 10) -> list[int]: ...


class SimpleBPE:
    """Small deterministic BPE for datasets whose raw text is anonymized integer terms."""

    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {"[UNK]": 0}
        self.merges: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
        self.merge_ranks: dict[tuple[tuple[str, ...], tuple[str, ...]], int] = {}

    @staticmethod
    def _symbol_key(symbol: tuple[str, ...]) -> str:
        return " ".join(symbol)

    @staticmethod
    def _symbol_from_key(value: str) -> tuple[str, ...]:
        return tuple(value.split(" ")) if value else ()

    def train(
        self,
        sequences: Iterable[Sequence[int]],
        *,
        vocabulary_size: int,
        minimum_frequency: int = 2,
        maximum_sequences: int = 0,
    ) -> int:
        corpus: Counter[tuple[tuple[str, ...], ...]] = Counter()
        base_symbols: set[tuple[str, ...]] = set()
        seen = 0
        for sequence in sequences:
            if maximum_sequences and seen >= maximum_sequences:
                break
            symbols = tuple((str(int(token)),) for token in sequence)
            if not symbols:
                continue
            corpus[symbols] += 1
            base_symbols.update(symbols)
            seen += 1
        for symbol in sorted(base_symbols):
            key = self._symbol_key(symbol)
            if key not in self.vocabulary:
                self.vocabulary[key] = len(self.vocabulary)

        while len(self.vocabulary) < vocabulary_size:
            pair_counts: Counter[tuple[tuple[str, ...], tuple[str, ...]]] = Counter()
            for symbols, frequency in corpus.items():
                for index in range(len(symbols) - 1):
                    pair_counts[(symbols[index], symbols[index + 1])] += frequency
            if not pair_counts:
                break
            pair, frequency = min(
                pair_counts.items(),
                key=lambda item: (-item[1], self._symbol_key(item[0][0]), self._symbol_key(item[0][1])),
            )
            if frequency < minimum_frequency:
                break
            left, right = pair
            merged = left + right
            self.merges.append(pair)
            self.merge_ranks[pair] = len(self.merges) - 1
            key = self._symbol_key(merged)
            if key not in self.vocabulary:
                self.vocabulary[key] = len(self.vocabulary)

            updated: Counter[tuple[tuple[str, ...], ...]] = Counter()
            for symbols, count in corpus.items():
                output: list[tuple[str, ...]] = []
                index = 0
                while index < len(symbols):
                    if index + 1 < len(symbols) and symbols[index] == left and symbols[index + 1] == right:
                        output.append(merged)
                        index += 2
                    else:
                        output.append(symbols[index])
                        index += 1
                updated[tuple(output)] += count
            corpus = updated
        return seen

    def encode_tokens(self, tokens: Sequence[int]) -> list[int]:
        symbols = [(str(int(token)),) for token in tokens]
        while True:
            selected_index = -1
            selected_rank = math.inf
            for index in range(len(symbols) - 1):
                rank = self.merge_ranks.get((symbols[index], symbols[index + 1]))
                if rank is not None and rank < selected_rank:
                    selected_index = index
                    selected_rank = rank
            if selected_index < 0:
                break
            symbols[selected_index : selected_index + 2] = [
                symbols[selected_index] + symbols[selected_index + 1]
            ]
        return [self.vocabulary.get(self._symbol_key(symbol), 0) for symbol in symbols]

    def save(self, path: str | Path) -> None:
        resolved = Path(path).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "lgmps.integer-bpe.v1",
            "vocabulary": self.vocabulary,
            "merges": [[self._symbol_key(left), self._symbol_key(right)] for left, right in self.merges],
        }
        with resolved.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> SimpleBPE:
        resolved = reject_lfs_pointer(path)
        with resolved.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("format") != "lgmps.integer-bpe.v1":
            raise ValueError(f"unsupported integer BPE file: {resolved}")
        tokenizer = cls()
        tokenizer.vocabulary = {str(key): int(value) for key, value in payload["vocabulary"].items()}
        tokenizer.merges = [
            (cls._symbol_from_key(left), cls._symbol_from_key(right)) for left, right in payload["merges"]
        ]
        tokenizer.merge_ranks = {pair: index for index, pair in enumerate(tokenizer.merges)}
        return tokenizer


class IntegerQueryEncoder:
    def __init__(self, bpe: SimpleBPE | None = None) -> None:
        self.bpe = bpe

    def encode(self, value: object, *, maximum_tokens: int = 10) -> list[int]:
        tokens = parse_integer_tokens(value)
        if self.bpe is not None:
            tokens = self.bpe.encode_tokens(tokens)
        return tokens[:maximum_tokens]


class HashSmokeTextEncoder:
    """Deterministic fixture encoder; deliberately named so it cannot look production-grade."""

    def __init__(self, vocabulary_size: int = 151_936) -> None:
        if vocabulary_size < 2:
            raise ValueError("vocabulary_size must be at least two")
        self.vocabulary_size = int(vocabulary_size)

    @staticmethod
    def _pieces(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+|[\u3400-\u9fff]|[^\s]", text.casefold())

    def encode(self, value: object, *, maximum_tokens: int = 10) -> list[int]:
        output: list[int] = []
        for piece in self._pieces(str(value))[:maximum_tokens]:
            digest = hashlib.blake2b(piece.encode("utf-8"), digest_size=8).digest()
            output.append(int.from_bytes(digest, "big") % (self.vocabulary_size - 1))
        return output


class HuggingFaceQueryEncoder:
    def __init__(self, model_or_path: str) -> None:
        if not model_or_path:
            raise ValueError("a tokenizer model or local path is required")
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise ImportError("the Hugging Face query backend requires transformers") from error
        self.tokenizer = AutoTokenizer.from_pretrained(model_or_path, trust_remote_code=True)

    def encode(self, value: object, *, maximum_tokens: int = 10) -> list[int]:
        return [
            int(token)
            for token in self.tokenizer.encode(
                str(value),
                add_special_tokens=False,
                truncation=True,
                max_length=maximum_tokens,
            )
        ]


def stable_text_embedding(text: str, *, dimension: int) -> list[float]:
    if dimension <= 0:
        raise ValueError("embedding dimension must be positive")
    vector = [0.0] * dimension
    pieces = HashSmokeTextEncoder._pieces(str(text))
    for piece in pieces:
        digest = hashlib.blake2b(piece.encode("utf-8"), digest_size=16).digest()
        index = int.from_bytes(digest[:8], "big") % dimension
        vector[index] += 1.0 if digest[8] & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector
