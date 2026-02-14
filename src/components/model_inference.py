"""
Component 4 – Model Inference

Wraps:  src/run_inference.py  →  InferencePipeline (renamed to avoid clash)
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.entity.config_entity import ModelInferenceConfig, PipelineConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelInferenceArtifact,
)

logger = logging.getLogger(__name__)


class ModelInference:
    """Run batch inference with the pretrained model."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        config: ModelInferenceConfig,
        transformation_artifact: DataTransformationArtifact,
    ):
        self.pipeline_config = pipeline_config
        self.config = config
        self.transformation_artifact = transformation_artifact

    # ------------------------------------------------------------------ #
    def initiate_model_inference(self) -> ModelInferenceArtifact:
        logger.info("=" * 60)
        logger.info("STAGE 4 — Model Inference")
        logger.info("=" * 60)

        from src.run_inference import (
            InferencePipeline as _InferencePipeline,
            InferenceConfig as _InferenceConfig,
        )

        # Build the internal config
        model_path = self.config.model_path or (
            self.pipeline_config.models_pretrained_dir
            / "fine_tuned_model_1dcnnbilstm.keras"
        )
        input_npy = self.config.input_npy or self.transformation_artifact.production_X_path
        output_dir = Path(
            self.config.output_dir
            or self.pipeline_config.data_prepared_dir / "predictions"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        inf_cfg = _InferenceConfig(
            model_path=Path(model_path),
            input_path=Path(input_npy),
            output_dir=output_dir,
            batch_size=self.config.batch_size,
            confidence_threshold=self.config.confidence_threshold,
            mode=self.config.mode,
        )

        pipe = _InferencePipeline(config=inf_cfg)
        t0 = time.time()
        result = pipe.run()
        inference_time = time.time() - t0

        # Parse result dict to artifact
        # run() returns {"results": DataFrame, "output_files": {"csv": Path, "npy": Path, "json": Path}, "probabilities": ndarray}
        output_files = result.get("output_files", {})
        predictions_csv = Path(
            output_files.get("csv", output_dir / "predictions_fresh.csv")
        )
        probabilities_npy = output_files.get("npy")
        if probabilities_npy:
            probabilities_npy = Path(probabilities_npy)

        # Predictions npy (labels) is derived from csv name
        predictions_npy = output_dir / "production_predictions_fresh.npy"

        # Extract real metrics from the results DataFrame
        results_df = result.get("results")
        n_predictions = len(results_df) if results_df is not None else 0

        # Extract model version from config
        model_version = ""
        try:
            inf_config = result.get("config", {})
            model_version = str(inf_config.get("model_path", model_path))
            # Use just the filename as version identifier
            model_version = Path(model_version).stem
        except Exception:
            pass

        # Extract activity distribution and confidence stats for the artifact
        activity_distribution = {}
        confidence_stats = {}
        if results_df is not None and len(results_df) > 0:
            try:
                if "predicted_activity" in results_df.columns:
                    activity_distribution = results_df["predicted_activity"].value_counts().to_dict()
                elif "predicted_class" in results_df.columns:
                    activity_distribution = results_df["predicted_class"].value_counts().to_dict()
            except Exception:
                pass
            try:
                if "confidence" in results_df.columns:
                    conf = results_df["confidence"]
                    confidence_stats = {
                        "mean": float(conf.mean()),
                        "std": float(conf.std()),
                        "min": float(conf.min()),
                        "max": float(conf.max()),
                        "median": float(conf.median()),
                    }
                    if "confidence_level" in results_df.columns:
                        confidence_stats["levels"] = results_df["confidence_level"].value_counts().to_dict()
                    if "is_uncertain" in results_df.columns:
                        confidence_stats["n_uncertain"] = int(results_df["is_uncertain"].sum())
            except Exception:
                pass

        logger.info(f"Inference complete: {n_predictions} predictions in {inference_time:.2f}s")
        if activity_distribution:
            logger.info(f"Activity distribution: {activity_distribution}")
        if confidence_stats:
            logger.info(f"Mean confidence: {confidence_stats.get('mean', 0):.3f}")

        return ModelInferenceArtifact(
            predictions_csv_path=predictions_csv,
            predictions_npy_path=predictions_npy,
            probabilities_npy_path=probabilities_npy,
            n_predictions=n_predictions,
            inference_time_seconds=inference_time,
            model_version=model_version,
            activity_distribution=activity_distribution,
            confidence_stats=confidence_stats,
        )
