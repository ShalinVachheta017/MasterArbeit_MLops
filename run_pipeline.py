#!/usr/bin/env python3
"""
=============================================================================
HAR MLOps — Production Pipeline  (Single Entry Point)
=============================================================================

Run the ENTIRE 14-stage production pipeline with one command:

    python run_pipeline.py                             # stages 1-7
    python run_pipeline.py --retrain --adapt adabn     # + AdaBN retraining
    python run_pipeline.py --advanced                  # + calibration, drift, etc.
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
  ── advanced analytics (--advanced) ──
    11  calibration       →  temperature scaling, MC Dropout, ECE
    12  wasserstein_drift →  Wasserstein distance, change-point detection
    13  curriculum_pseudo_labeling → progressive self-training with EWC
    14  sensor_placement  →  hand detection, axis mirroring augmentation

Result LOG → logs/pipeline/pipeline_result_<timestamp>.log
Result JSON → logs/pipeline/pipeline_result_<timestamp>.json

=============================================================================
"""

import sys
import argparse
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import centralized logger
from src.logger import logging, CURRENT_LOG_FILE
logger = logging.getLogger(__name__)

from src.pipeline.production_pipeline import ProductionPipeline
from src.entity.config_entity import (
    PipelineConfig,
    DataIngestionConfig,
    DataTransformationConfig,
    ModelInferenceConfig,
    ModelRetrainingConfig,
)

# Logging is already configured in src.logger module - no need for basicConfig here


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

  # Advanced analytics (calibration, Wasserstein drift, etc.)
  python run_pipeline.py --advanced

  # Curriculum pseudo-labeling only
  python run_pipeline.py --stages curriculum_pseudo_labeling --curriculum-iterations 10

  # Run specific advanced stages
  python run_pipeline.py --stages calibration wasserstein_drift sensor_placement

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
            "calibration", "wasserstein_drift",
            "curriculum_pseudo_labeling", "sensor_placement",
        ],
        default=None,
        help="Run only these stages (default: 1-7; use --retrain for 8-10, --advanced for 11-14)",
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
        "--config",
        type=str,
        default="config/pipeline_config.yaml",
        help="Path to YAML config file (default: config/pipeline_config.yaml)",
    )
    parser.add_argument(
        "--gravity-removal",
        action="store_true",
        default=None,
        help="Enable gravity removal (overrides config file)",
    )
    parser.add_argument(
        "--no-unit-conversion",
        action="store_true",
        help="Disable unit conversion milliG→m/s² (overrides config file)",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        default=None,
        help="Enable domain calibration (overrides config file)",
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

    # ── Advanced analytics flags ──────────────────────────────────────
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Include stages 11-14 (calibration, Wasserstein drift, "
             "curriculum pseudo-labeling, sensor placement)",
    )
    parser.add_argument(
        "--curriculum-iterations",
        type=int,
        default=5,
        help="Number of curriculum pseudo-labeling iterations (default: 5)",
    )
    parser.add_argument(
        "--ewc-lambda",
        type=float,
        default=1000.0,
        help="EWC regularization strength (default: 1000.0)",
    )
    parser.add_argument(
        "--mc-dropout-passes",
        type=int,
        default=30,
        help="Number of MC Dropout forward passes (default: 30)",
    )

    return parser.parse_args()


def load_preprocessing_config(config_path: str) -> dict:
    """Load preprocessing toggles from YAML config file."""
    import yaml
    path = Path(config_path)
    if not path.exists():
        print(f"Config file not found: {path}, using defaults")
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get('preprocessing', {})


def main():
    # Display startup banner with log file location
    print("=" * 80)
    print("HAR MLOps Production Pipeline")
    print("=" * 80)
    print(f"Log file: {CURRENT_LOG_FILE}")
    print("=" * 80)
    print()
    
    logger.info("Pipeline starting...")
    
    args = parse_args()

    # ── Load config from YAML ─────────────────────────────────────────
    yaml_preproc = load_preprocessing_config(args.config)

    # Resolve preprocessing toggles: CLI flags override YAML config
    enable_unit_conversion = yaml_preproc.get('enable_unit_conversion', True)
    enable_gravity_removal = yaml_preproc.get('enable_gravity_removal', False)
    enable_calibration = yaml_preproc.get('enable_calibration', False)

    # CLI overrides
    if args.no_unit_conversion:
        enable_unit_conversion = False
    if args.gravity_removal:
        enable_gravity_removal = True
    if args.calibrate:
        enable_calibration = True

    # Show what's active using logger
    logger.info("Preprocessing configuration (from %s):", args.config)
    logger.info("  Unit Conversion (milliG→m/s²): %s", 'ON' if enable_unit_conversion else 'OFF')
    logger.info("  Gravity Removal:               %s", 'ON' if enable_gravity_removal else 'OFF')
    logger.info("  Domain Calibration:            %s", 'ON' if enable_calibration else 'OFF')

    # ── Build configs ─────────────────────────────────────────────────
    pipeline_cfg = PipelineConfig()

    ingestion_cfg = DataIngestionConfig(
        input_csv=Path(args.input_csv) if args.input_csv else None,
    )

    transformation_cfg = DataTransformationConfig(
        enable_unit_conversion=enable_unit_conversion,
        enable_gravity_removal=enable_gravity_removal,
        enable_calibration=enable_calibration,
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

    # Advanced configs
    from src.entity.config_entity import (
        CalibrationUncertaintyConfig,
        WassersteinDriftConfig,
        CurriculumPseudoLabelingConfig,
        SensorPlacementConfig,
    )
    calibration_cfg = CalibrationUncertaintyConfig(
        mc_forward_passes=args.mc_dropout_passes,
    )
    wasserstein_cfg = WassersteinDriftConfig()
    curriculum_cfg = CurriculumPseudoLabelingConfig(
        n_iterations=args.curriculum_iterations,
        ewc_lambda=args.ewc_lambda,
    )
    sensor_cfg = SensorPlacementConfig()

    # ── Build and run pipeline ────────────────────────────────────────
    pipeline = ProductionPipeline(
        pipeline_config=pipeline_cfg,
        ingestion_config=ingestion_cfg,
        transformation_config=transformation_cfg,
        inference_config=inference_cfg,
        retraining_config=retraining_cfg,
        registration_config=registration_cfg,
        calibration_config=calibration_cfg,
        wasserstein_config=wasserstein_cfg,
        curriculum_config=curriculum_cfg,
        sensor_placement_config=sensor_cfg,
    )

    result = pipeline.run(
        stages=args.stages,
        skip_ingestion=args.skip_ingestion,
        skip_validation=args.skip_validation,
        continue_on_failure=args.continue_on_failure,
        enable_retrain=args.retrain,
    )

    # ── Print Summary ─────────────────────────────────────────────────
    _print_pipeline_summary(result)

    # ── Exit code ─────────────────────────────────────────────────────
    if result.overall_status == "SUCCESS":
        sys.exit(0)
    else:
        sys.exit(1)


def _print_pipeline_summary(result):
    """Print a clean summary of key pipeline metrics."""
    print("\n" + "="*70)
    print("  PIPELINE SUMMARY")
    print("="*70)
    
    # Overall status
    status_emoji = "✅" if result.overall_status == "SUCCESS" else "⚠️" if result.overall_status == "PARTIAL" else "❌"
    print(f"\n{status_emoji} Overall Status: {result.overall_status}")
    print(f"   Completed: {len(result.stages_completed)} stages")
    print(f"   Failed: {len(result.stages_failed)} stages")
    
    # Key metrics
    print("\n📊 KEY METRICS")
    print("-" * 70)
    
    # Inference metrics
    if result.inference:
        print(f"\n  Inference:")
        print(f"    • Predictions: {result.inference.n_predictions:,}")
        print(f"    • Duration: {result.inference.inference_time_seconds:.2f}s")
        if result.inference.confidence_stats:
            conf = result.inference.confidence_stats
            print(f"    • Mean Confidence: {conf.get('mean', 0)*100:.1f}%")
            print(f"    • Uncertain: {conf.get('n_uncertain', 0)} ({conf.get('n_uncertain', 0)/max(result.inference.n_predictions, 1)*100:.1f}%)")
    
    # Monitoring metrics
    if result.monitoring:
        print(f"\n  Monitoring:")
        print(f"    • Overall: {result.monitoring.overall_status}")
        
        if result.monitoring.layer3_drift:
            drift = result.monitoring.layer3_drift
            drift_score = drift.get('max_drift', 0)
            print(f"    • Drift Score: {drift_score:.4f}")
            
            # Drift interpretation (thresholds: 0.75 warn, 1.50 alert)
            if drift_score > 1.50:
                print(f"      ⚠️ HIGH DRIFT ({drift_score:.3f}) - Retraining recommended")
            elif drift_score > 0.75:
                print(f"      ⚠ Moderate drift ({drift_score:.3f}) - Monitor closely")
            else:
                print(f"      ✓ Drift within acceptable range")
    
    # Trigger decision
    if result.trigger:
        print(f"\n  Retraining Decision:")
        retrain_emoji = "🔄" if result.trigger.should_retrain else "✓"
        print(f"    {retrain_emoji} Should Retrain: {'YES' if result.trigger.should_retrain else 'NO'}")
        print(f"    • Alert Level: {result.trigger.alert_level}")
        if result.trigger.reasons:
            print(f"    • Reasons:")
            for reason in result.trigger.reasons[:3]:  # Show first 3
                print(f"      - {reason}")
    
    # Activity distribution
    if result.inference and result.inference.activity_distribution:
        print(f"\n  Top 3 Activities Detected:")
        sorted_activities = sorted(
            result.inference.activity_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        for activity, count in sorted_activities:
            pct = count / max(result.inference.n_predictions, 1) * 100
            print(f"    • {activity}: {count} ({pct:.1f}%)")
    
    print("\n" + "="*70)
    print(f"📁 Artifacts saved to: artifacts/{result.run_id}")
    print(f"📄 Log file: {CURRENT_LOG_FILE}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
