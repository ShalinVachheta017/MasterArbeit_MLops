"""
Component 3 – Data Transformation (Preprocessing)

Wraps:  src/preprocess_data.py  →  UnifiedPreprocessor
        (CSV → normalized, windowed .npy)
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
)
from src.entity.config_entity import DataTransformationConfig, PipelineConfig

logger = logging.getLogger(__name__)


class DataTransformation:
    """Preprocess the fused CSV into windowed .npy arrays."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        config: DataTransformationConfig,
        ingestion_artifact: DataIngestionArtifact,
        validation_artifact: Optional[DataValidationArtifact] = None,
    ):
        self.pipeline_config = pipeline_config
        self.config = config
        self.ingestion_artifact = ingestion_artifact
        self.validation_artifact = validation_artifact

    # ------------------------------------------------------------------ #
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logger.info("=" * 60)
        logger.info("STAGE 3 — Data Transformation")
        logger.info("=" * 60)

        from src.preprocess_data import PreprocessLogger, UnifiedPreprocessor

        # Determine input CSV
        csv_path = self.config.input_csv or self.ingestion_artifact.fused_csv_path
        csv_path = Path(csv_path)
        logger.info("Input CSV: %s", csv_path)

        output_dir = Path(self.config.output_dir or self.pipeline_config.data_prepared_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Read CSV
        df = pd.read_csv(csv_path)

        # Setup preprocessor
        prep_logger = PreprocessLogger("preprocessing").get_logger()
        preprocessor = UnifiedPreprocessor(
            logger=prep_logger,
            window_size=self.config.window_size,
            overlap=self.config.overlap,
        )

        # Detect format and sensor columns
        data_format, sensor_cols = preprocessor.detect_data_format(df)
        logger.info("Data format: %s  |  sensors: %s", data_format, sensor_cols)

        # Track which preprocessing steps were applied
        unit_conversion = False
        gravity_removal_applied = False
        calibration_applied = False

        # Step 1: Unit conversion (milliG → m/s²) — independent of gravity removal
        if (
            self.config.enable_unit_conversion
            or self.config.enable_gravity_removal
            or self.config.enable_calibration
        ):
            from src.preprocess_data import UnitDetector

            detector = UnitDetector(prep_logger)
            accel_cols = [c for c in sensor_cols if c.startswith(("Ax", "Ay", "Az"))]
            df, unit_conversion = detector.process_units(df, accel_cols)

        # Step 2: Gravity removal (only if explicitly enabled)
        if self.config.enable_gravity_removal:
            from src.preprocess_data import GravityRemover

            gravity = GravityRemover(prep_logger)
            df = gravity.remove_gravity(df, enable=True)
            gravity_removal_applied = True

        # Step 3: Domain calibration (only if explicitly enabled)
        if self.config.enable_calibration:
            from src.preprocess_data import DomainCalibrator

            calibrator = DomainCalibrator(prep_logger)
            df = calibrator.calibrate(df, enable=True)
            calibration_applied = True

        # Normalize
        df = preprocessor.normalize_data(df, sensor_cols, mode="transform")

        # Create windows
        X, y, metadata = preprocessor.create_windows(df, sensor_cols, data_format)
        logger.info("Windows created: X shape=%s", X.shape)

        # Save
        np.save(output_dir / "production_X.npy", X)
        import json

        meta_path = output_dir / "production_metadata.json"
        serialisable_meta = []
        for m in metadata:
            item = {}
            for k, v in m.items():
                if isinstance(v, np.integer):
                    item[k] = int(v)
                elif isinstance(v, np.floating):
                    item[k] = float(v)
                else:
                    item[k] = v
            serialisable_meta.append(item)
        with open(meta_path, "w") as f:
            json.dump(serialisable_meta, f, indent=2, default=str)

        if y is not None:
            np.save(output_dir / "production_y.npy", y)
            logger.info("Labels saved: y shape=%s", y.shape)

        return DataTransformationArtifact(
            production_X_path=output_dir / "production_X.npy",
            metadata_path=meta_path,
            n_windows=X.shape[0],
            window_size=X.shape[1],
            unit_conversion_applied=unit_conversion,
            gravity_removal_applied=gravity_removal_applied,
            calibration_applied=calibration_applied,
            preprocessing_timestamp=datetime.now().isoformat(),
        )
