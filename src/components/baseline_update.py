"""    
Component 10 – Baseline Update

Wraps:  scripts/build_training_baseline.py  →  BaselineBuilder
"""

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.entity.config_entity import BaselineUpdateConfig, PipelineConfig
from src.entity.artifact_entity import (
    ModelRetrainingArtifact,
    BaselineUpdateArtifact,
)

logger = logging.getLogger(__name__)


class BaselineUpdate:
    """Rebuild drift baselines after retraining."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        config: BaselineUpdateConfig,
        retraining_artifact: Optional[ModelRetrainingArtifact] = None,
    ):
        self.pipeline_config = pipeline_config
        self.config = config
        self.retraining_artifact = retraining_artifact

    # ------------------------------------------------------------------ #
    def initiate_baseline_update(self) -> BaselineUpdateArtifact:
        logger.info("=" * 60)
        logger.info("STAGE 10 — Baseline Update")
        logger.info("=" * 60)

        import sys
        scripts_dir = str(self.pipeline_config.scripts_dir)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        from build_training_baseline import BaselineBuilder

        builder = BaselineBuilder()

        # Determine data source for baseline
        data_path = self.config.training_data_path or (
            self.pipeline_config.data_raw_dir.parent / "all_users_data_labeled.csv"
        )
        logger.info("Building baseline from: %s", data_path)

        # Build
        baseline = builder.build_from_csv(data_path)

        # Save paths
        output_baseline = self.config.output_baseline_path or (
            self.pipeline_config.models_dir / "training_baseline.json"
        )
        output_normalized = self.config.output_normalized_path or (
            self.pipeline_config.models_dir / "normalized_baseline.json"
        )

        promote = getattr(self.config, "promote_to_shared", False)

        if promote:
            # Write to shared paths that monitoring reads at runtime.
            # Also save a versioned copy so the change is reversible.
            builder.save(output_baseline)
            logger.info("Baseline PROMOTED to shared path: %s", output_baseline)

            builder.save_normalized(output_normalized)
            logger.info("Normalised baseline PROMOTED to shared path: %s", output_normalized)

            # Versioned archive (e.g. training_baseline_20260219_130000.json)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            versioned_dir = Path(output_baseline).parent / "baseline_versions"
            versioned_dir.mkdir(parents=True, exist_ok=True)
            for src_p, dst_name in [
                (output_baseline,   f"training_baseline_{ts}.json"),
                (output_normalized, f"normalized_baseline_{ts}.json"),
            ]:
                shutil.copy2(src_p, versioned_dir / dst_name)
                logger.info("Versioned archive saved: %s", versioned_dir / dst_name)
        else:
            # Governance default: do NOT overwrite shared baseline.
            # Only save to artifact dir and log to MLflow.
            logger.info(
                "Baseline NOT promoted to shared path (promote_to_shared=False). "
                "Re-run with --update-baseline to promote."
            )
            # Still build in-memory so we can copy the files into the artifact dir.
            builder.save(output_baseline)       # writes to output_baseline path
            builder.save_normalized(output_normalized)

        # Copy baselines into artifact dir for traceability (always)
        artifact_models = Path(self.pipeline_config.artifact_dir) / "models"
        artifact_models.mkdir(parents=True, exist_ok=True)
        for src_file in (output_baseline, output_normalized):
            src = Path(src_file)
            if src.exists():
                dst = artifact_models / src.name
                shutil.copy2(src, dst)
                logger.info("Baseline copied to artifact: %s", dst)
            else:
                logger.debug("Baseline file not on disk yet (mock run?), skipping artifact copy: %s", src)

        # Log baseline files as MLflow artifacts if a run is active
        try:
            import mlflow
            if mlflow.active_run():
                mlflow.log_artifact(str(artifact_models / Path(output_baseline).name), artifact_path="baseline")
                mlflow.log_artifact(str(artifact_models / Path(output_normalized).name), artifact_path="baseline")
                mlflow.log_param("baseline_promoted", promote)
                logger.info("Baseline files logged as MLflow artifacts.")
        except Exception as _mlflow_err:
            logger.debug("MLflow baseline artifact logging skipped: %s", _mlflow_err)

        # Stats summary
        stats = {
            "n_channels": baseline.get("n_channels", 6),
            "n_samples": baseline.get("n_samples", 0),
            "activities": list(baseline.get("per_class", {}).keys()) if "per_class" in baseline else [],
        }

        return BaselineUpdateArtifact(
            baseline_path=Path(output_baseline),
            normalized_baseline_path=Path(output_normalized),
            n_channels=stats["n_channels"],
            stats_summary=stats,
            update_timestamp=datetime.now().isoformat(),
        )
