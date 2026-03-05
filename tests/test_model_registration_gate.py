"""
P1-4: Model Registration Gate Tests
======================================
Verify that model_registration.py correctly blocks or allows deployment
based on val_accuracy comparison + degradation_tolerance.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.components.model_registration import ModelRegistration
from src.entity.artifact_entity import ModelRegistrationArtifact, ModelRetrainingArtifact
from src.entity.config_entity import ModelRegistrationConfig, PipelineConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CURRENT_ACC = 0.90  # accuracy of the "deployed" model in all tests


def _fake_registry_patch(current_acc: float = _CURRENT_ACC):
    """Return patch context managers that simulate an existing registry entry."""
    return (
        patch(
            "src.model_rollback.ModelRegistry.get_current_version",
            return_value="v1.0.0",
        ),
        patch(
            "src.model_rollback.ModelRegistry.list_versions",
            return_value=[{"version": "v1.0.0", "metrics": {"val_accuracy": current_acc}}],
        ),
        patch(
            "src.model_rollback.ModelRegistry.register_model",
            return_value=MagicMock(version="v1.0.1"),
        ),
        patch(
            "src.model_rollback.ModelRegistry.deploy_model",
            return_value=False,
        ),
    )


def _make_comp(
    tmp_path: Path,
    new_accuracy,
    degradation_tolerance: float = 0.005,
    block_if_no_metrics: bool = False,
    metrics_override: dict = None,
) -> ModelRegistration:
    """Build a ModelRegistration component with the given accuracy."""
    fake_model = tmp_path / "model.keras"
    fake_model.write_text("fake")

    metrics = {"val_accuracy": new_accuracy} if new_accuracy is not None else {}
    if metrics_override is not None:
        metrics = metrics_override

    retrain = ModelRetrainingArtifact(
        retrained_model_path=fake_model,
        training_report={},
        adaptation_method="standard",
        metrics=metrics,
        n_target_samples=100,
        retraining_timestamp="2026-02-26T00:00:00",
    )
    cfg = ModelRegistrationConfig(
        proxy_validation=True,
        auto_deploy=False,
        degradation_tolerance=degradation_tolerance,
        block_if_no_metrics=block_if_no_metrics,
    )
    pipeline_cfg = PipelineConfig(project_root=tmp_path)
    return ModelRegistration(pipeline_cfg, cfg, retrain)


def _run(comp: ModelRegistration) -> ModelRegistrationArtifact:
    patches = _fake_registry_patch()
    with patches[0], patches[1], patches[2], patches[3]:
        return comp.initiate_model_registration()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_improved_model_is_accepted(tmp_path):
    """New model 0.92 > current 0.90 → is_better=True."""
    result = _run(_make_comp(tmp_path, new_accuracy=0.92))
    assert result.is_better_than_current is True


def test_equal_accuracy_is_accepted(tmp_path):
    """New model 0.90 == current 0.90 → is_better=True (within tolerance)."""
    result = _run(_make_comp(tmp_path, new_accuracy=0.90))
    assert result.is_better_than_current is True


def test_small_regression_within_tolerance_is_accepted(tmp_path):
    """0.897 vs 0.90 → delta=0.003 < tol=0.005 → is_better=True."""
    result = _run(_make_comp(tmp_path, new_accuracy=0.897, degradation_tolerance=0.005))
    assert result.is_better_than_current is True


def test_large_regression_outside_tolerance_is_blocked(tmp_path):
    """0.88 vs 0.90 → delta=0.02 > tol=0.005 → is_better=False → deployment blocked."""
    result = _run(_make_comp(tmp_path, new_accuracy=0.88, degradation_tolerance=0.005))
    assert result.is_better_than_current is False


def test_zero_tolerance_blocks_any_drop(tmp_path):
    """With tolerance=0.0, even a 0.001 drop is blocked."""
    result = _run(_make_comp(tmp_path, new_accuracy=0.899, degradation_tolerance=0.0))
    assert result.is_better_than_current is False


def test_tta_model_without_metrics_defaults_to_true(tmp_path):
    """AdaBN/TENT unsupervised TTA: no val_accuracy → is_better=True, not deployed."""
    result = _run(
        _make_comp(
            tmp_path,
            new_accuracy=None,
            block_if_no_metrics=False,
            metrics_override={"after_mean_confidence": 0.85},
        )
    )
    assert result.is_better_than_current is True
    assert result.is_deployed is False  # auto_deploy=False


def test_tta_model_with_block_if_no_metrics_is_blocked(tmp_path):
    """When block_if_no_metrics=True and TTA model has no val_accuracy → blocked."""
    result = _run(
        _make_comp(
            tmp_path,
            new_accuracy=None,
            block_if_no_metrics=True,
            metrics_override={"after_mean_confidence": 0.85},
        )
    )
    assert result.is_better_than_current is False
