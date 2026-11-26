from __future__ import annotations

import pytest

from lgmps.inference.metrics import RetrievalMetrics


def test_paper_recall_preserves_relevant_item_multiplicity() -> None:
    first = (1, 257, 513)
    second = (2, 258, 514)
    missed = (3, 259, 515)
    metrics = RetrievalMetrics(cutoffs=(1, 2))
    metrics.update(
        [first, second],
        [first, first, missed],
        token_length=100,
        session_count=3,
    )
    summary = metrics.summary()

    assert summary["paper_recall_at_1"] == pytest.approx(2 / 3)
    assert summary["paper_recall_at_2"] == pytest.approx(2 / 3)
    assert summary["unique_sid_coverage_at_1"] == pytest.approx(1 / 2)
    assert summary["sample_recall_at_1"] == 1.0


def test_paper_recall_reports_macro_and_micro_aggregation() -> None:
    hit = (1, 257, 513)
    miss = (2, 258, 514)
    metrics = RetrievalMetrics(cutoffs=(1,))
    metrics.update([hit], [hit], token_length=100, session_count=3)
    metrics.update([miss], [hit, hit, hit], token_length=100, session_count=3)
    summary = metrics.summary()

    assert summary["paper_recall_at_1"] == pytest.approx(0.5)
    assert summary["paper_recall_micro_at_1"] == pytest.approx(0.25)


def test_paper_metrics_reject_duplicate_beams() -> None:
    prediction = (1, 257, 513)
    metrics = RetrievalMetrics(cutoffs=(1,))
    with pytest.raises(ValueError, match="unique SID predictions"):
        metrics.update(
            [prediction, prediction],
            [prediction],
            token_length=100,
            session_count=3,
        )
