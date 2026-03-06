from __future__ import annotations

import pytest

from src.utils.temporal_metrics import flip_rate_per_session, summarize_rates


def test_flip_rate_per_session_uses_adjacent_windows_in_order():
    labels = [0, 0, 1, 1, 1, 2, 2]
    session_ids = ["A", "A", "A", "A", "B", "B", "B"]
    timestamps = [0, 1, 2, 3, 0, 1, 2]

    rates = flip_rate_per_session(labels, session_ids, timestamps)

    # Session A: transitions 0->0->1->1 => 1 flip over 3 adjacent pairs
    assert rates["A"] == pytest.approx(1.0 / 3.0)
    # Session B: transitions 1->2->2 => 1 flip over 2 adjacent pairs
    assert rates["B"] == pytest.approx(0.5)


def test_flip_rate_rejects_shuffled_windows():
    labels = [0, 1, 1, 0]
    session_ids = ["S1", "S1", "S1", "S1"]
    # Timestamp goes backwards (2 -> 1), indicating shuffled windows.
    timestamps = [0, 2, 1, 3]

    with pytest.raises(ValueError, match="not sorted within session"):
        flip_rate_per_session(labels, session_ids, timestamps)


def test_flip_rate_summary_returns_median_and_p95():
    summary = summarize_rates({"S1": 0.10, "S2": 0.40, "S3": 0.30})

    assert summary["n_sessions"] == 3
    assert summary["median"] == pytest.approx(0.30)
    assert summary["p95"] == pytest.approx(0.39, abs=0.02)
