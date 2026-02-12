#!/usr/bin/env python3
"""
=============================================================================
HAR MLOps — Production Pipeline  (Single Entry Point)
=============================================================================

Run the ENTIRE 10-stage production pipeline with one command:

    python run_pipeline.py                             # stages 1-7
    python run_pipeline.py --retrain --adapt adabn     # + AdaBN retraining
    python run_pipeline.py --stages inference evaluation  # specific stages
    python run_pipeline.py --input-csv my_recording.csv   # your own data

Pipeline stages (in order):
     1  ingestion         →  raw Garmin Excel / CSV → sensor_fused_50Hz.csv
     2  validation        →  schema + value-range checks
     3  transformation    →  CSV → normalised, windowed production_X.npy
     4  inference         →  .npy + pretrained model → predictions CSV/NPY
     5  evaluation        →  confidence / distribution / ECE analysis
     6  monitoring        →  3-layer (confidence, temporal, drift)
     7  trigger           →  automated retraining decision
  ── retraining cycle (--retrain) ──
     8  retraining        →  standard / AdaBN / pseudo-label
     9  registration      →  version, deploy, rollback
    10  baseline_update   →  rebuild drift baselines

Result JSON → logs/pipeline/pipeline_result_<timestamp>.json

=============================================================================
"""

import sys
import logging
import argparse
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.pipeline.production_pipeline import ProductionPipeline
from src.entity.config_entity import (
    PipelineConfig,
    DataIngestionConfig,
    DataTransformationConfig,
    ModelInferenceConfig,
    ModelRetrainingConfig,
)

# ── Logging setup ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="HAR MLOps Production Pipeline — run all stages with one command",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full inference cycle (stages 1-7)
  python run_pipeline.py

  # Skip ingestion — CSV already exists
  python run_pipeline.py --skip-ingestion

  # Inference onward (stages 4-7)
  python run_pipeline.py --stages inference evaluation monitoring trigger

  # Your own recording — CSV input
  python run_pipeline.py --input-csv data/raw/my_recording.csv

  # Full cycle + AdaBN domain adaptation
  python run_pipeline.py --retrain --adapt adabn

  # Retrain with pseudo-labeling
  python run_pipeline.py --retrain --adapt pseudo_label

  # Continue past errors
  python run_pipeline.py --continue-on-failure
        """,
    )

    parser.add_argument(
        "--stages",
        nargs="+",
        choices=[
            "ingestion", "validation", "transformation",
            "inference", "evaluation", "monitoring", "trigger",
            "retraining", "registration", "baseline_update",
        ],
        default=None,
        help="Run only these stages (default: 1-7; use --retrain for 8-10)",
    )
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip data ingestion — use existing sensor_fused_50Hz.csv",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip data validation",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Log errors and continue to next stage instead of aborting",
    )

    # ── Input overrides ───────────────────────────────────────────────
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Path to fused CSV (your own recording, ABCD cases)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to .keras model for inference",
    )
    parser.add_argument(
        "--gravity-removal",
        action="store_true",
        help="Enable gravity removal during preprocessing",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Enable domain calibration during preprocessing",
    )

    # ── Retraining flags ──────────────────────────────────────────────
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Include stages 8-10 (retraining → registration → baseline)",
    )
    parser.add_argument(
        "--adapt",
        type=str,
        choices=["adabn", "pseudo_label", "none"],
        default="none",
        help="Domain adaptation method for retraining (default: none)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Path to ground-truth labels for supervised retraining",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs for retraining (default: 100)",
    )
    parser.add_argument(
        "--auto-deploy",
        action="store_true",
        help="Automatically deploy retrained model if proxy validation passes",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Build configs ─────────────────────────────────────────────────
    pipeline_cfg = PipelineConfig()

    ingestion_cfg = DataIngestionConfig(
        input_csv=Path(args.input_csv) if args.input_csv else None,
    )

    transformation_cfg = DataTransformationConfig(
        enable_gravity_removal=args.gravity_removal,
        enable_calibration=args.calibrate,
    )

    inference_cfg = ModelInferenceConfig(
        model_path=Path(args.model) if args.model else None,
    )

    retraining_cfg = ModelRetrainingConfig(
        enable_adaptation=(args.adapt != "none"),
        adaptation_method=args.adapt,
        epochs=args.epochs,
        labels_path=Path(args.labels) if args.labels else None,
    )

    from src.entity.config_entity import ModelRegistrationConfig
    registration_cfg = ModelRegistrationConfig(
        auto_deploy=args.auto_deploy,
    )

    # ── Build and run pipeline ────────────────────────────────────────
    pipeline = ProductionPipeline(
        pipeline_config=pipeline_cfg,
        ingestion_config=ingestion_cfg,
        transformation_config=transformation_cfg,
        inference_config=inference_cfg,
        retraining_config=retraining_cfg,
        registration_config=registration_cfg,
    )

    result = pipeline.run(
        stages=args.stages,
        skip_ingestion=args.skip_ingestion,
        skip_validation=args.skip_validation,
        continue_on_failure=args.continue_on_failure,
        enable_retrain=args.retrain,
    )

    # ── Exit code ─────────────────────────────────────────────────────
    if result.overall_status == "SUCCESS":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
