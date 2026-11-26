from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def _average_precision(
    predictions: list[tuple[int, ...]], targets: set[tuple[int, ...]], cutoff: int
) -> float:
    if not targets:
        return 0.0
    hits = 0
    precision_sum = 0.0
    seen: set[tuple[int, ...]] = set()
    for rank, prediction in enumerate(predictions[:cutoff], start=1):
        if prediction in seen:
            continue
        seen.add(prediction)
        if prediction in targets:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(len(targets), cutoff)


def _mean_and_se(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": float("nan"), "standard_error": float("nan")}
    mean = sum(values) / len(values)
    if len(values) == 1:
        standard_error = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        standard_error = math.sqrt(variance / len(values))
    return {"count": len(values), "mean": mean, "standard_error": standard_error}


class RetrievalMetrics:
    def __init__(self, cutoffs: tuple[int, ...] = (1, 5, 20)) -> None:
        self.cutoffs = cutoffs
        self.examined = 0
        self.eligible = 0
        self.sample_hits = {cutoff: 0 for cutoff in cutoffs}
        self.item_recall_sum = {cutoff: 0.0 for cutoff in cutoffs}
        self.item_hits = {cutoff: 0 for cutoff in cutoffs}
        self.unique_sid_coverage_sum = {cutoff: 0.0 for cutoff in cutoffs}
        self.relevant_items = 0
        self.average_precision_sum = 0.0
        self.strata: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    @staticmethod
    def _length_bucket(length: int) -> str:
        for upper in (512, 1024, 2048, 4096):
            if length <= upper:
                return f"<= {upper}"
        return "> 4096"

    @staticmethod
    def _session_bucket(sessions: int) -> str:
        for upper in (5, 10, 20, 50, 100, 256):
            if sessions <= upper:
                return f"<= {upper}"
        return "> 256"

    def update(
        self,
        predictions: list[tuple[int, ...]],
        targets: list[tuple[int, ...]],
        *,
        token_length: int,
        session_count: int,
    ) -> None:
        self.examined += 1
        if len(predictions) != len(set(predictions)):
            raise ValueError("Paper evaluation requires an ordered list of unique SID predictions")
        target_set = set(targets)
        if not target_set:
            return
        self.eligible += 1
        self.relevant_items += len(targets)
        per_sample: dict[str, float] = {}
        for cutoff in self.cutoffs:
            predicted = set(predictions[:cutoff])
            hit = float(bool(predicted & target_set))
            item_hits = sum(target in predicted for target in targets)
            item_recall = item_hits / len(targets)
            unique_sid_coverage = len(predicted & target_set) / len(target_set)
            self.sample_hits[cutoff] += int(hit)
            self.item_recall_sum[cutoff] += item_recall
            self.item_hits[cutoff] += item_hits
            self.unique_sid_coverage_sum[cutoff] += unique_sid_coverage
            per_sample[f"sample_recall_at_{cutoff}"] = hit
            per_sample[f"paper_recall_at_{cutoff}"] = item_recall
            per_sample[f"unique_sid_coverage_at_{cutoff}"] = unique_sid_coverage
        average_precision = _average_precision(predictions, target_set, max(self.cutoffs))
        self.average_precision_sum += average_precision
        per_sample[f"map_at_{max(self.cutoffs)}"] = average_precision
        for family, bucket in (
            ("by_token_length", self._length_bucket(token_length)),
            ("by_session_count", self._session_bucket(session_count)),
        ):
            for name, value in per_sample.items():
                self.strata[f"{family}:{bucket}"][name].append(value)

    def summary(self) -> dict[str, Any]:
        denominator = max(self.eligible, 1)
        metrics: dict[str, Any] = {
            "examined_samples": self.examined,
            "eligible_samples": self.eligible,
            "definitions": {
                "sample_recall_at_k": "fraction of samples with at least one target SID in the top-k beams",
                "paper_recall_at_k": (
                    "macro mean, over eligible requests, of the fraction of relevant items whose SID is in "
                    "the top-k unique beams; repeated target SIDs retain item multiplicity"
                ),
                "unique_sid_coverage_at_k": "mean fraction of unique relevant SIDs covered by the top-k beams",
                "map_at_k": "deduplicated average precision over unique target SIDs",
            },
        }
        for cutoff in self.cutoffs:
            metrics[f"sample_recall_at_{cutoff}"] = self.sample_hits[cutoff] / denominator
            metrics[f"paper_recall_at_{cutoff}"] = self.item_recall_sum[cutoff] / denominator
            metrics[f"paper_recall_micro_at_{cutoff}"] = self.item_hits[cutoff] / max(self.relevant_items, 1)
            metrics[f"unique_sid_coverage_at_{cutoff}"] = self.unique_sid_coverage_sum[cutoff] / denominator
        metrics[f"map_at_{max(self.cutoffs)}"] = self.average_precision_sum / denominator
        stratified: dict[str, Any] = {}
        for key, values_by_metric in sorted(self.strata.items()):
            family, bucket = key.split(":", 1)
            stratified.setdefault(family, {})[bucket] = {
                name: _mean_and_se(values) for name, values in values_by_metric.items()
            }
        metrics["stratified"] = stratified
        return metrics
