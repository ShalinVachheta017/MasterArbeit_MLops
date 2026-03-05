"""
tests/test_config_loader.py
============================
Unit tests for src/utils/config_loader.py (P0-2 coverage).

Implements three tests from reports/PIPELINE_CTO_REVIEW.md:
  1. load_yaml_overrides() applies override values correctly.
  2. load_yaml_overrides() with a missing file → returns empty dict (no crash).
  3. apply_overrides() silently ignores unknown keys (no exception, no side-effect).
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.entity.config_entity import PostInferenceMonitoringConfig, TriggerEvaluationConfig
from src.utils.config_loader import apply_overrides, load_yaml_overrides


class TestLoadYamlOverrides:
    """load_yaml_overrides() must parse YAML and return a usable dict."""

    def test_valid_yaml_returns_expected_values(self, tmp_path):
        """Override values in the YAML are returned correctly."""
        yaml_file = tmp_path / "overrides.yaml"
        yaml_file.write_text(
            textwrap.dedent("""\
                monitoring:
                  confidence_warn_threshold: 0.55
                  drift_zscore_threshold: 2.5
                trigger:
                  cooldown_hours: 12
            """),
            encoding="utf-8",
        )

        result = load_yaml_overrides(path=yaml_file)

        assert result["monitoring"]["confidence_warn_threshold"] == pytest.approx(0.55)
        assert result["monitoring"]["drift_zscore_threshold"] == pytest.approx(2.5)
        assert result["trigger"]["cooldown_hours"] == 12

    def test_missing_file_returns_empty_dict(self, tmp_path):
        """A non-existent path must not raise — returns {} so defaults are used."""
        missing = tmp_path / "does_not_exist.yaml"
        result = load_yaml_overrides(path=missing)
        assert result == {}

    def test_empty_yaml_returns_empty_dict(self, tmp_path):
        """An empty (or whitespace-only) YAML file returns {} without error."""
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("", encoding="utf-8")
        result = load_yaml_overrides(path=empty_file)
        assert result == {}

    def test_env_var_path_is_used(self, tmp_path, monkeypatch):
        """HAR_PIPELINE_OVERRIDES env var is respected when no path is given."""
        yaml_file = tmp_path / "env_overrides.yaml"
        yaml_file.write_text(
            "monitoring:\n  max_baseline_age_days: 60\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HAR_PIPELINE_OVERRIDES", str(yaml_file))

        result = load_yaml_overrides()  # no explicit path

        assert result["monitoring"]["max_baseline_age_days"] == 60


class TestApplyOverrides:
    """apply_overrides() must patch dataclass fields and ignore unknowns."""

    def test_known_field_is_overridden(self):
        """A valid field name is updated on the config instance."""
        cfg = PostInferenceMonitoringConfig()
        original = cfg.confidence_warn_threshold

        apply_overrides(cfg, {"confidence_warn_threshold": 0.50})

        assert cfg.confidence_warn_threshold == pytest.approx(0.50)
        # Confirms the value actually changed
        assert cfg.confidence_warn_threshold != original

    def test_unknown_key_is_silently_ignored(self):
        """An unknown key must NOT raise and must NOT corrupt other fields."""
        cfg = PostInferenceMonitoringConfig()
        before = cfg.drift_zscore_threshold

        # Should not raise; unknown key must be silently skipped
        apply_overrides(cfg, {"this_key_does_not_exist": 999})

        assert cfg.drift_zscore_threshold == pytest.approx(before)

    def test_multiple_fields_applied_at_once(self):
        """Multiple overrides in a single section are all applied."""
        cfg = TriggerEvaluationConfig()

        apply_overrides(
            cfg,
            {
                "confidence_warn": 0.70,
                "cooldown_hours": 48,
            },
        )

        assert cfg.confidence_warn == pytest.approx(0.70)
        assert cfg.cooldown_hours == 48

    def test_empty_section_is_no_op(self):
        """An empty override section leaves the config unchanged."""
        cfg = PostInferenceMonitoringConfig()
        before_threshold = cfg.confidence_warn_threshold

        apply_overrides(cfg, {})

        assert cfg.confidence_warn_threshold == pytest.approx(before_threshold)
