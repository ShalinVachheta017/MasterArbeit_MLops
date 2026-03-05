"""Utilities for strict temporal metrics computed on ordered session streams."""

from __future__ import annotations

from typing import Dict, Hashable, Iterable, List

import numpy as np


def _validate_shapes(
    labels: np.ndarray, session_ids: np.ndarray, timestamps: np.ndarray
) -> None:
    if len(labels) != len(session_ids) or len(labels) != len(timestamps):
        raise ValueError(
            "labels, session_ids, and timestamps must have the same length "
            f"(got {len(labels)}, {len(session_ids)}, {len(timestamps)})."
        )


def _validate_strict_session_order(session_ids: np.ndarray, timestamps: np.ndarray) -> None:
    """Ensure records are grouped by session and strictly ordered by timestamp."""
    if len(session_ids) == 0:
        return

    seen_sessions = set()
    current_session = session_ids[0]
    last_timestamp = timestamps[0]

    for idx in range(1, len(session_ids)):
        sid = session_ids[idx]
        ts = timestamps[idx]

        if sid != current_session:
            seen_sessions.add(current_session)
            if sid in seen_sessions:
                raise ValueError(
                    "Session records are not contiguous. Windows appear to be shuffled "
                    f"(session '{sid}' reappeared)."
                )
            current_session = sid
            last_timestamp = ts
            continue

        if ts < last_timestamp:
            raise ValueError(
                "Timestamps are not sorted within session. Flip-rate must be computed "
                "on adjacent windows in timestamp order."
            )
        last_timestamp = ts


def flip_rate_per_session(
    labels: Iterable[int | str],
    session_ids: Iterable[Hashable],
    timestamps: Iterable[float | int],
) -> Dict[Hashable, float]:
    """Compute strict per-session flip rate.

    flip_rate(session) = (# label changes between adjacent windows in timestamp order)
                         / (n_windows_in_session - 1)
    """
    labels_arr = np.asarray(list(labels))
    sessions_arr = np.asarray(list(session_ids))
    times_arr = np.asarray(list(timestamps), dtype=float)

    _validate_shapes(labels_arr, sessions_arr, times_arr)
    _validate_strict_session_order(sessions_arr, times_arr)

    out: Dict[Hashable, float] = {}
    if len(labels_arr) == 0:
        return out

    start = 0
    for idx in range(1, len(labels_arr) + 1):
        is_boundary = idx == len(labels_arr) or sessions_arr[idx] != sessions_arr[start]
        if not is_boundary:
            continue

        sid = sessions_arr[start]
        sess_labels = labels_arr[start:idx]
        if len(sess_labels) <= 1:
            out[sid] = 0.0
        else:
            flips = np.count_nonzero(sess_labels[1:] != sess_labels[:-1])
            out[sid] = float(flips / (len(sess_labels) - 1))
        start = idx

    return out


def summarize_rates(rates: Dict[Hashable, float]) -> Dict[str, float]:
    """Aggregate session-level rates for reporting."""
    if not rates:
        return {"median": 0.0, "p95": 0.0, "n_sessions": 0}

    vals = np.array(list(rates.values()), dtype=float)
    return {
        "median": float(np.median(vals)),
        "p95": float(np.percentile(vals, 95)),
        "n_sessions": int(vals.size),
    }

