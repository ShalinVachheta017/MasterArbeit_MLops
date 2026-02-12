"""
=============================================================================
HAR MLOps — Production Pipeline Orchestrator
=============================================================================

Clean ~200-line orchestrator that delegates ALL work to component classes.
Follows the reference pattern from vikashishere/YT-MLops-Proj1.

10 Stages:
    1  Data Ingestion           (Excel/CSV → fused CSV)
    2  Data Validation          (schema + range checks)
    3  Data Transformation      (CSV → windowed .npy)
    4  Model Inference          (.npy + model → predictions)
    5  Model Evaluation         (confidence / distribution / ECE)
    6  Post-Inference Monitoring (3-layer: confidence, temporal, drift)
    7  Trigger Evaluation       (retraining decision)
  ── retraining cycle (optional, stages 8-10) ──
    8  Model Retraining         (standard / AdaBN / pseudo-label)
    9  Model Registration       (version, deploy, rollback)
   10  Baseline Update          (rebuild drift baselines)
"""

import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.entity.config_entity import (
    PipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelInferenceConfig,
    ModelEvaluationConfig,
    PostInferenceMonitoringConfig,
    TriggerEvaluationConfig,
    ModelRetrainingConfig,
    ModelRegistrationConfig,
    BaselineUpdateConfig,
)
from src.entity.artifact_entity import PipelineResult

logger = logging.getLogger(__name__)

# Ordered list of ALL stages
ALL_STAGES = [
    "ingestion", "validation", "transformation",
    "inference", "evaluation", "monitoring", "trigger",
    "retraining", "registration", "baseline_update",
]
RETRAIN_STAGES = {"retraining", "registration", "baseline_update"}


class ProductionPipeline:
    """Orchestrates the full HAR MLOps pipeline."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        *,
        ingestion_config: Optional[DataIngestionConfig] = None,
        validation_config: Optional[DataValidationConfig] = None,
        transformation_config: Optional[DataTransformationConfig] = None,
        inference_config: Optional[ModelInferenceConfig] = None,
        evaluation_config: Optional[ModelEvaluationConfig] = None,
        monitoring_config: Optional[PostInferenceMonitoringConfig] = None,
        trigger_config: Optional[TriggerEvaluationConfig] = None,
        retraining_config: Optional[ModelRetrainingConfig] = None,
        registration_config: Optional[ModelRegistrationConfig] = None,
        baseline_config: Optional[BaselineUpdateConfig] = None,
    ):
        self.pipeline_config = pipeline_config
        self.ingestion_config = ingestion_config or DataIngestionConfig()
        self.validation_config = validation_config or DataValidationConfig()
        self.transformation_config = transformation_config or DataTransformationConfig()
        self.inference_config = inference_config or ModelInferenceConfig()
        self.evaluation_config = evaluation_config or ModelEvaluationConfig()
        self.monitoring_config = monitoring_config or PostInferenceMonitoringConfig()
        self.trigger_config = trigger_config or TriggerEvaluationConfig()
        self.retraining_config = retraining_config or ModelRetrainingConfig()
        self.registration_config = registration_config or ModelRegistrationConfig()
        self.baseline_config = baseline_config or BaselineUpdateConfig()

    # ================================================================== #
    def run(
        self,
        stages: Optional[List[str]] = None,
        skip_ingestion: bool = False,
        skip_validation: bool = False,
        continue_on_failure: bool = False,
        enable_retrain: bool = False,
    ) -> PipelineResult:
        """
        Execute the pipeline.

        Parameters
        ----------
        stages : list[str], optional
            Subset of stages to run.  Default → stages 1-7;
            add "retraining" etc. for the retrain cycle.
        skip_ingestion : bool
            Skip stage 1.
        skip_validation : bool
            Skip stage 2.
        continue_on_failure : bool
            Log errors and continue to next stage instead of aborting.
        enable_retrain : bool
            Include stages 8-10 even when `stages` is None.
        """
        result = PipelineResult(
            run_id=self.pipeline_config.timestamp,
            start_time=datetime.now().isoformat(),
        )

        # Determine which stages to run
        if stages is not None:
            run_stages = [s for s in ALL_STAGES if s in stages]
        else:
            run_stages = [s for s in ALL_STAGES if s not in RETRAIN_STAGES]
            if enable_retrain:
                run_stages.extend(s for s in ALL_STAGES if s in RETRAIN_STAGES)

        if skip_ingestion and "ingestion" in run_stages:
            run_stages.remove("ingestion")
            result.stages_skipped.append("ingestion")
        if skip_validation and "validation" in run_stages:
            run_stages.remove("validation")
            result.stages_skipped.append("validation")

        logger.info("Pipeline stages: %s", run_stages)

        # Holders for cross-stage artifacts
        ingestion_art = None
        validation_art = None
        transformation_art = None
        inference_art = None
        evaluation_art = None
        monitoring_art = None
        trigger_art = None
        retraining_art = None
        registration_art = None
        baseline_art = None

        # MLflow tracking (optional — wraps entire run)
        mlflow_tracker = self._init_mlflow()

        for stage in run_stages:
            try:
                if stage == "ingestion":
                    from src.components.data_ingestion import DataIngestion
                    comp = DataIngestion(self.pipeline_config, self.ingestion_config)
                    ingestion_art = comp.initiate_data_ingestion()
                    result.ingestion = ingestion_art

                elif stage == "validation":
                    if ingestion_art is None:
                        ingestion_art = self._make_fallback_ingestion_artifact()
                    from src.components.data_validation import DataValidation
                    comp = DataValidation(self.pipeline_config, self.validation_config, ingestion_art)
                    validation_art = comp.initiate_data_validation()
                    result.validation = validation_art
                    if not validation_art.is_valid:
                        logger.warning("Validation FAILED — errors: %s", validation_art.errors)
                        if not continue_on_failure:
                            result.stages_failed.append("validation")
                            break

                elif stage == "transformation":
                    if ingestion_art is None:
                        ingestion_art = self._make_fallback_ingestion_artifact()
                    from src.components.data_transformation import DataTransformation
                    comp = DataTransformation(
                        self.pipeline_config, self.transformation_config,
                        ingestion_art, validation_art,
                    )
                    transformation_art = comp.initiate_data_transformation()
                    result.transformation = transformation_art

                elif stage == "inference":
                    if transformation_art is None:
                        transformation_art = self._make_fallback_transformation_artifact()
                    from src.components.model_inference import ModelInference
                    comp = ModelInference(
                        self.pipeline_config, self.inference_config, transformation_art,
                    )
                    inference_art = comp.initiate_model_inference()
                    result.inference = inference_art

                elif stage == "evaluation":
                    if inference_art is None:
                        raise ValueError("No inference artifact — run inference first.")
                    from src.components.model_evaluation import ModelEvaluation
                    comp = ModelEvaluation(
                        self.pipeline_config, self.evaluation_config, inference_art,
                    )
                    evaluation_art = comp.initiate_model_evaluation()
                    result.evaluation = evaluation_art

                elif stage == "monitoring":
                    if inference_art is None:
                        raise ValueError("No inference artifact — run inference first.")
                    from src.components.post_inference_monitoring import PostInferenceMonitoring
                    comp = PostInferenceMonitoring(
                        self.pipeline_config, self.monitoring_config,
                        inference_art, transformation_art,
                    )
                    monitoring_art = comp.initiate_post_inference_monitoring()
                    result.monitoring = monitoring_art

                elif stage == "trigger":
                    if monitoring_art is None:
                        raise ValueError("No monitoring artifact — run monitoring first.")
                    from src.components.trigger_evaluation import TriggerEvaluation
                    comp = TriggerEvaluation(
                        self.pipeline_config, self.trigger_config, monitoring_art,
                    )
                    trigger_art = comp.initiate_trigger_evaluation()
                    result.trigger = trigger_art

                elif stage == "retraining":
                    from src.components.model_retraining import ModelRetraining
                    comp = ModelRetraining(
                        self.pipeline_config, self.retraining_config,
                        trigger_art, transformation_art,
                    )
                    retraining_art = comp.initiate_model_retraining()
                    result.retraining = retraining_art

                elif stage == "registration":
                    if retraining_art is None:
                        raise ValueError("No retraining artifact — run retraining first.")
                    from src.components.model_registration import ModelRegistration
                    comp = ModelRegistration(
                        self.pipeline_config, self.registration_config,
                        retraining_art, evaluation_art,
                    )
                    registration_art = comp.initiate_model_registration()
                    result.registration = registration_art

                elif stage == "baseline_update":
                    from src.components.baseline_update import BaselineUpdate
                    comp = BaselineUpdate(
                        self.pipeline_config, self.baseline_config, retraining_art,
                    )
                    baseline_art = comp.initiate_baseline_update()
                    result.baseline_update = baseline_art

                result.stages_completed.append(stage)
                logger.info("✓ Stage '%s' completed.", stage)

                # Log stage metrics to MLflow
                if mlflow_tracker:
                    self._log_stage_to_mlflow(mlflow_tracker, stage, result)

            except Exception as e:
                logger.error("✗ Stage '%s' FAILED: %s", stage, e)
                logger.debug(traceback.format_exc())
                result.stages_failed.append(stage)
                if not continue_on_failure:
                    break

        # Finalise
        result.end_time = datetime.now().isoformat()
        result.overall_status = (
            "SUCCESS" if not result.stages_failed
            else "PARTIAL" if result.stages_completed
            else "FAILED"
        )
        self._save_result(result)

        if mlflow_tracker:
            self._end_mlflow(mlflow_tracker, result)

        logger.info("=" * 60)
        logger.info("Pipeline finished: %s  (completed=%d  failed=%d  skipped=%d)",
                     result.overall_status,
                     len(result.stages_completed),
                     len(result.stages_failed),
                     len(result.stages_skipped))
        logger.info("=" * 60)

        return result

    # ================================================================== #
    # Helpers
    # ================================================================== #

    def _make_fallback_ingestion_artifact(self):
        """Create a synthetic ingestion artifact pointing to existing CSV."""
        from src.entity.artifact_entity import DataIngestionArtifact
        csv = self.pipeline_config.data_processed_dir / "sensor_fused_50Hz.csv"
        if not csv.exists():
            csv = self.ingestion_config.input_csv or csv
        return DataIngestionArtifact(
            fused_csv_path=Path(csv),
            n_rows=0, n_columns=0, sampling_hz=50,
            ingestion_timestamp=datetime.now().isoformat(),
            source_type="fallback",
        )

    def _make_fallback_transformation_artifact(self):
        """Create a synthetic transformation artifact pointing to existing .npy."""
        from src.entity.artifact_entity import DataTransformationArtifact
        npy = self.pipeline_config.data_prepared_dir / "production_X.npy"
        return DataTransformationArtifact(
            production_X_path=npy,
            metadata_path=self.pipeline_config.data_prepared_dir / "production_metadata.json",
            n_windows=0, window_size=200,
            unit_conversion_applied=False,
            preprocessing_timestamp=datetime.now().isoformat(),
        )

    def _save_result(self, result: PipelineResult):
        """Persist pipeline result as JSON."""
        log_dir = self.pipeline_config.logs_dir / "pipeline"
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"pipeline_result_{self.pipeline_config.timestamp}.json"

        # Convert dataclasses to dicts
        import dataclasses
        data = dataclasses.asdict(result)
        # Pathlib → str
        def _convert(obj):
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert(i) for i in obj]
            return obj

        with open(path, "w") as f:
            json.dump(_convert(data), f, indent=2, default=str)
        logger.info("Pipeline result saved: %s", path)

    # ── MLflow helpers ─────────────────────────────────────────────────
    def _init_mlflow(self):
        try:
            from src.mlflow_tracking import MLflowTracker
            tracker = MLflowTracker(experiment_name="har-production-pipeline")
            tracker.__enter_run = tracker.start_run(
                run_name=f"pipeline_{self.pipeline_config.timestamp}",
                tags={"pipeline": "production", "version": "2.0"},
            )
            tracker._ctx = tracker.__enter_run.__enter__()
            return tracker
        except Exception as e:
            logger.debug("MLflow not available, skipping: %s", e)
            return None

    def _log_stage_to_mlflow(self, tracker, stage, result):
        try:
            metrics = {}
            if stage == "inference" and result.inference:
                metrics["inference_n_predictions"] = result.inference.n_predictions
                metrics["inference_time_seconds"] = result.inference.inference_time_seconds
            elif stage == "monitoring" and result.monitoring:
                metrics["monitoring_status"] = 1.0 if result.monitoring.overall_status == "HEALTHY" else 0.0
            elif stage == "retraining" and result.retraining:
                for k, v in result.retraining.metrics.items():
                    if isinstance(v, (int, float)):
                        metrics[f"retrain_{k}"] = v
            if metrics:
                tracker.log_metrics(metrics)
        except Exception:
            pass

    def _end_mlflow(self, tracker, result):
        try:
            tracker.log_params({
                "stages_completed": ",".join(result.stages_completed),
                "stages_failed": ",".join(result.stages_failed),
                "overall_status": result.overall_status,
            })
            tracker.__enter_run.__exit__(None, None, None)
        except Exception:
            pass
