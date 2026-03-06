"""
tests/test_validation_gate.py
==============================
Verify that the production pipeline validation gate always raises
DataValidationError when data is invalid — even when continue_on_failure=True.

Implements P0-1 test coverage from reports/PIPELINE_CTO_REVIEW.md.

Gap fixed: production_pipeline.py:271-275 used to `break` silently when
continue_on_failure=False.  Now it unconditionally raises DataValidationError.
"""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.entity.config_entity import PipelineConfig
from src.entity.artifact_entity import DataValidationArtifact
from src.exceptions import DataValidationError
from src.pipeline.production_pipeline import ProductionPipeline


def _make_pipeline() -> ProductionPipeline:
    """Return a pipeline with minimal configuration."""
    return ProductionPipeline(pipeline_config=PipelineConfig())


def _invalid_artifact() -> DataValidationArtifact:
    """Simulate a failed validation result."""
    return DataValidationArtifact(
        is_valid=False,
        errors=["Expected 9 sensor columns, got 3"],
        warnings=[],
    )


class TestValidationGate:
    """Validation gate must hard-stop on invalid data regardless of CLI flags."""

    @patch("src.components.data_validation.DataValidation")
    def test_invalid_data_raises_error_default(self, mock_cls):
        """DataValidationError raised with default continue_on_failure=False."""
        mock_comp = MagicMock()
        mock_comp.initiate_data_validation.return_value = _invalid_artifact()
        mock_cls.return_value = mock_comp

        pipeline = _make_pipeline()
        with pytest.raises(DataValidationError) as exc_info:
            pipeline.run(stages=["validation"])

        assert (
            "error(s)" in str(exc_info.value).lower()
            or "validation failed" in str(exc_info.value).lower()
        )

    @patch("src.components.data_validation.DataValidation")
    def test_invalid_data_raises_error_with_continue_on_failure(self, mock_cls):
        """DataValidationError MUST also raise when continue_on_failure=True.

        This is the key regression guard: before the fix, the invalidation would
        silently `break` the stage loop instead of raising.
        """
        mock_comp = MagicMock()
        mock_comp.initiate_data_validation.return_value = _invalid_artifact()
        mock_cls.return_value = mock_comp

        pipeline = _make_pipeline()
        # continue_on_failure=True must NOT suppress DataValidationError
        with pytest.raises(DataValidationError):
            pipeline.run(stages=["validation"], continue_on_failure=True)

    @patch("src.components.data_validation.DataValidation")
    def test_valid_data_does_not_raise(self, mock_cls):
        """A passing validation must not raise DataValidationError."""
        mock_comp = MagicMock()
        mock_comp.initiate_data_validation.return_value = DataValidationArtifact(
            is_valid=True,
            errors=[],
            warnings=["low_sample_count"],
        )
        mock_cls.return_value = mock_comp

        pipeline = _make_pipeline()
        # Should run without raising DataValidationError (may raise other errors
        # for missing downstream artifacts, but NOT DataValidationError)
        try:
            pipeline.run(stages=["validation"])
        except DataValidationError:
            pytest.fail("Valid data must not raise DataValidationError")
        except Exception:
            # Other pipeline exceptions (missing files, etc.) are acceptable here
            pass
