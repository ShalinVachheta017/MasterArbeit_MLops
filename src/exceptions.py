"""
src/exceptions.py
=================
Custom exception classes for the HAR MLOps pipeline.
"""


class DataValidationError(RuntimeError):
    """Raised when Stage 2 data validation fails.

    This is a hard stop — invalid data must never reach Stage 3
    (DataTransformation) regardless of the ``continue_on_failure`` flag.

    To handle this programmatically::

        from src.exceptions import DataValidationError
        try:
            pipeline.run()
        except DataValidationError as exc:
            print("Validation failed:", exc)
    """


class ModelLoadError(RuntimeError):
    """Raised when Stage 4 cannot load or verify the pretrained model."""


class ConfigurationError(ValueError):
    """Raised when a pipeline configuration is invalid or self-contradictory."""
