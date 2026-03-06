"""
tests/test_baseline_age_gauge.py
=================================
Unit tests for the har_baseline_age_days Prometheus gauge logic (P0-3 coverage).

The actual update runs inside src/api/app.py::_run_monitoring() inside an
``if _PROM_AVAILABLE:`` block.  These tests verify the *age-calculation formula*
and the *-1 sentinel* logic directly, without requiring a running FastAPI
server or Prometheus client.

From PIPELINE_CTO_REVIEW.md P0-3 tests:
  - file exists  → age in fractional days >= 0 (and close to 0 for a fresh file)
  - file missing → age == -1
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Age-calculation logic (mirrors app.py::_run_monitoring)
# ---------------------------------------------------------------------------


def _compute_baseline_age(baseline_path: Path) -> float:
    """Mirror of the gauge logic in src/api/app.py _run_monitoring().

    Returns fractional days since last modification, or -1 if file is missing.
    """
    if baseline_path.exists():
        return (time.time() - baseline_path.stat().st_mtime) / 86400
    return -1


class TestBaselineAgeCalculation:
    """Age formula: (now - mtime) / 86400, or -1 when missing."""

    def test_age_is_non_negative_for_fresh_file(self, tmp_path):
        """A freshly created file should have age >= 0 and < 1 day."""
        baseline = tmp_path / "normalized_baseline.json"
        baseline.write_text('{"mean": [], "std": []}', encoding="utf-8")

        age = _compute_baseline_age(baseline)

        assert age >= 0, "Age of existing file must be >= 0"
        assert age < 1.0, "Age of a just-created file must be < 1 day"

    def test_age_is_minus_one_when_file_missing(self, tmp_path):
        """A non-existent baseline path must yield the -1 sentinel."""
        missing = tmp_path / "normalized_baseline.json"
        assert not missing.exists(), "Precondition: file must not exist"

        age = _compute_baseline_age(missing)

        assert age == -1, "Missing baseline must return -1 (disabled-detection sentinel)"

    def test_age_reflects_known_mtime(self, tmp_path):
        """Manually set mtime 5 days ago → age should be approximately 5 days."""
        baseline = tmp_path / "normalized_baseline.json"
        baseline.write_text("{}", encoding="utf-8")

        # Set mtime 5 days in the past
        five_days_ago = time.time() - (5 * 86400)
        os.utime(baseline, (five_days_ago, five_days_ago))

        age = _compute_baseline_age(baseline)

        # Allow ±0.01 day (≈14 min) tolerance for test execution time
        assert abs(age - 5.0) < 0.01, f"Expected ~5 days, got {age:.4f}"


class TestBaselineAgeGaugeIntegration:
    """Verify app.py exposes the gauge and that BASELINE_PATH is wired correctly."""

    def test_baseline_path_constant_is_defined(self):
        """app.py must expose BASELINE_PATH pointing to models/normalized_baseline.json."""
        from src.api import app as _app

        assert hasattr(_app, "BASELINE_PATH"), "BASELINE_PATH must be defined in app.py"
        assert _app.BASELINE_PATH.name == "normalized_baseline.json"

    def test_prom_gauge_is_defined_when_prometheus_available(self):
        """_prom_baseline_age_days Gauge must exist in app module (if prometheus is installed)."""
        from src.api import app as _app

        if not getattr(_app, "_PROM_AVAILABLE", False):
            pytest.skip("prometheus_client not installed — gauge not defined")

        assert hasattr(_app, "_prom_baseline_age_days"), (
            "_prom_baseline_age_days Gauge missing from app.py — "
            "HARStaleDriftBaseline / HARMissingDriftBaseline alerts will have no data"
        )

    @patch("src.api.app.BASELINE_PATH")
    def test_run_monitoring_sets_gauge_minus_one_when_missing(self, mock_path):
        """When BASELINE_PATH.exists() is False, gauge must be set to -1."""
        from src.api import app as _app

        if not getattr(_app, "_PROM_AVAILABLE", False):
            pytest.skip("prometheus_client not installed")

        # Mock BASELINE_PATH.exists() → False
        mock_path.exists.return_value = False

        recorded = []
        with patch.object(_app._prom_baseline_age_days, "set", side_effect=recorded.append):
            import numpy as np

            # Minimal valid inputs for _run_monitoring
            n = 10
            n_classes = 6
            probs = np.ones((n, n_classes)) / n_classes
            predictions = np.zeros(n, dtype=int)
            windows = np.random.rand(n, 128, 9)
            _app._run_monitoring(predictions, probs, windows)

        assert (
            -1 in recorded
        ), "When baseline is missing, _prom_baseline_age_days.set(-1) must be called"
