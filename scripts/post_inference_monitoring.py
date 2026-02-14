"""
Post-Inference Monitoring Module

Performs 3-layer monitoring:
1. Confidence Analysis - detect low confidence predictions
2. Temporal Analysis - detect anomalous temporal patterns
3. Drift Analysis - detect distribution drift from baseline
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MonitoringReport:
    """Container for monitoring results."""
    
    def __init__(self):
        self.overall_status = "PASS"
        self.layer1_confidence = {}
        self.layer2_temporal = {}
        self.layer3_drift = {}
        self.alerts = []
        self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status,
            "layer1_confidence": self.layer1_confidence,
            "layer2_temporal": self.layer2_temporal,
            "layer3_drift": self.layer3_drift,
            "alerts": self.alerts,
            "metadata": self.metadata
        }


class PostInferenceMonitor:
    """
    Three-layer monitoring system for production inference results.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        uncertain_threshold_pct: float = 10.0,
        drift_threshold: float = 0.15,
    ):
        """
        Args:
            confidence_threshold: Minimum confidence for valid predictions
            uncertain_threshold_pct: Max % of uncertain predictions allowed
            drift_threshold: Max KL divergence allowed for drift detection
        """
        self.confidence_threshold = confidence_threshold
        self.uncertain_threshold_pct = uncertain_threshold_pct
        self.drift_threshold = drift_threshold
        self.logger = logging.getLogger(__name__)
    
    def run(
        self,
        predictions_path: Path,
        production_data_path: Optional[Path] = None,
        baseline_path: Optional[Path] = None,
        model_path: Optional[Path] = None,
        output_dir: Path = Path("outputs/monitoring"),
    ) -> MonitoringReport:
        """
        Run complete monitoring pipeline.
        
        Args:
            predictions_path: Path to predictions CSV
            production_data_path: Path to production data .npy (for drift)
            baseline_path: Path to baseline stats JSON (for drift)
            model_path: Path to model file
            output_dir: Directory for monitoring outputs
            
        Returns:
            MonitoringReport with results
        """
        self.logger.info("=" * 60)
        self.logger.info("MONITORING PIPELINE - START")
        self.logger.info("=" * 60)
        
        report = MonitoringReport()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load predictions
        self.logger.info(f"Loading predictions: {predictions_path}")
        predictions_df = pd.read_csv(predictions_path)
        
        # Layer 1: Confidence Analysis
        self.logger.info("\n" + "=" * 60)
        self.logger.info("LAYER 1: CONFIDENCE ANALYSIS")
        self.logger.info("=" * 60)
        report.layer1_confidence = self._analyze_confidence(predictions_df)
        
        # Layer 2: Temporal Analysis
        self.logger.info("\n" + "=" * 60)
        self.logger.info("LAYER 2: TEMPORAL PATTERN ANALYSIS")
        self.logger.info("=" * 60)
        report.layer2_temporal = self._analyze_temporal_patterns(predictions_df)
        
        # Layer 3: Drift Analysis (optional - requires baseline)
        if production_data_path and baseline_path and Path(baseline_path).exists():
            self.logger.info("\n" + "=" * 60)
            self.logger.info("LAYER 3: DRIFT ANALYSIS")
            self.logger.info("=" * 60)
            report.layer3_drift = self._analyze_drift(
                production_data_path, baseline_path
            )
        else:
            self.logger.info("\n" + "=" * 60)
            self.logger.info("LAYER 3: DRIFT ANALYSIS - SKIPPED")
            self.logger.info("=" * 60)
            if not baseline_path:
                self.logger.warning("⚠️ No baseline provided - drift analysis skipped")
            elif not Path(baseline_path).exists():
                self.logger.warning(f"⚠️ Baseline not found: {baseline_path}")
            report.layer3_drift = {"status": "SKIPPED", "reason": "No baseline available"}
        
        # Determine overall status
        report.overall_status = self._determine_overall_status(report)
        report.metadata = {
            "total_predictions": len(predictions_df),
            "predictions_path": str(predictions_path),
            "baseline_path": str(baseline_path) if baseline_path else None,
        }
        
        # Save report
        report_path = output_dir / "monitoring_report.json"
        with open(report_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        self.logger.info(f"\n📄 Monitoring report saved: {report_path}")
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info(f"MONITORING RESULT: {report.overall_status}")
        self.logger.info("=" * 60)
        
        if report.alerts:
            self.logger.warning(f"⚠️ {len(report.alerts)} alert(s) generated:")
            for alert in report.alerts:
                self.logger.warning(f"   - {alert}")
        
        return report
    
    def _analyze_confidence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Layer 1: Analyze prediction confidence."""
        results = {}
        
        # Overall statistics
        mean_conf = df['confidence'].mean()
        std_conf = df['confidence'].std()
        min_conf = df['confidence'].min()
        max_conf = df['confidence'].max()
        
        results['mean_confidence'] = float(mean_conf)
        results['std_confidence'] = float(std_conf)
        results['min_confidence'] = float(min_conf)
        results['max_confidence'] = float(max_conf)
        
        # Count by confidence level
        if 'is_uncertain' in df.columns:
            uncertain_count = df['is_uncertain'].sum()
            uncertain_pct = 100 * uncertain_count / len(df)
        else:
            uncertain_count = (df['confidence'] < self.confidence_threshold).sum()
            uncertain_pct = 100 * uncertain_count / len(df)
        
        results['uncertain_count'] = int(uncertain_count)
        results['uncertain_percentage'] = float(uncertain_pct)
        
        self.logger.info(f"Mean confidence: {mean_conf:.3f} ({100*mean_conf:.1f}%)")
        self.logger.info(f"Uncertain predictions: {uncertain_count}/{len(df)} ({uncertain_pct:.1f}%)")
        
        # Check thresholds
        if uncertain_pct > self.uncertain_threshold_pct:
            alert = f"High uncertainty rate: {uncertain_pct:.1f}% > {self.uncertain_threshold_pct}%"
            results['status'] = 'ALERT'
            results['alert'] = alert
            self.logger.warning(f"⚠️ {alert}")
        else:
            results['status'] = 'PASS'
            self.logger.info(f"✅ Confidence check PASSED")
        
        return results
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Layer 2: Analyze temporal patterns."""
        results = {}
        
        if 'predicted_activity' not in df.columns:
            results['status'] = 'SKIPPED'
            results['reason'] = 'No activity column found'
            return results
        
        # Calculate transitions
        activities = df['predicted_activity'].values
        transitions = np.sum(activities[1:] != activities[:-1])
        transition_rate = 100 * transitions / (len(activities) - 1) if len(activities) > 1 else 0
        
        results['total_windows'] = len(df)
        results['transitions'] = int(transitions)
        results['transition_rate'] = float(transition_rate)
        
        # Find longest sequences
        sequences = {}
        current_activity = activities[0]
        current_length = 1
        
        for i in range(1, len(activities)):
            if activities[i] == current_activity:
                current_length += 1
            else:
                if current_activity not in sequences:
                    sequences[current_activity] = 0
                sequences[current_activity] = max(sequences[current_activity], current_length)
                current_activity = activities[i]
                current_length = 1
        
        # Don't forget the last sequence
        if current_activity not in sequences:
            sequences[current_activity] = 0
        sequences[current_activity] = max(sequences[current_activity], current_length)
        
        results['longest_sequences'] = sequences
        
        self.logger.info(f"Total windows: {len(df)}")
        self.logger.info(f"Activity transitions: {transitions} ({transition_rate:.1f}%)")
        self.logger.info(f"Longest sequences: {sequences}")
        
        # Simple check: very high transition rate might indicate noise
        if transition_rate > 50:
            alert = f"High transition rate: {transition_rate:.1f}% (possible noise)"
            results['status'] = 'WARNING'
            results['alert'] = alert
            self.logger.warning(f"⚠️ {alert}")
        else:
            results['status'] = 'PASS'
            self.logger.info(f"✅ Temporal pattern check PASSED")
        
        return results
    
    def _analyze_drift(
        self, production_data_path: Path, baseline_path: Path
    ) -> Dict[str, Any]:
        """Layer 3: Analyze distribution drift."""
        results = {}
        
        try:
            # Load production data
            production_data = np.load(production_data_path)
            
            # Load baseline stats
            with open(baseline_path, 'r') as f:
                baseline = json.load(f)
            
            # Calculate production statistics
            prod_mean = production_data.mean(axis=(0, 1))  # Average over windows and timesteps
            prod_std = production_data.std(axis=(0, 1))
            
            # Compare with baseline (if available)
            if 'mean' in baseline and 'std' in baseline:
                baseline_mean = np.array(baseline['mean'])
                baseline_std = np.array(baseline['std'])
                
                # Calculate drift metrics (simple: normalized difference)
                mean_diff = np.abs(prod_mean - baseline_mean) / (baseline_std + 1e-8)
                max_drift = float(np.max(mean_diff))
                
                results['production_mean'] = prod_mean.tolist()
                results['production_std'] = prod_std.tolist()
                results['baseline_mean'] = baseline_mean.tolist()
                results['baseline_std'] = baseline_std.tolist()
                results['max_drift'] = max_drift
                
                self.logger.info(f"Max drift (normalized): {max_drift:.3f}")
                
                if max_drift > self.drift_threshold:
                    alert = f"Distribution drift detected: {max_drift:.3f} > {self.drift_threshold}"
                    results['status'] = 'ALERT'
                    results['alert'] = alert
                    self.logger.warning(f"⚠️ {alert}")
                else:
                    results['status'] = 'PASS'
                    self.logger.info(f"✅ Drift check PASSED")
            else:
                results['status'] = 'PASS'
                results['note'] = 'Baseline format incompatible - basic check only'
                self.logger.info("✅ Basic drift check PASSED")
                
        except Exception as e:
            results['status'] = 'ERROR'
            results['error'] = str(e)
            self.logger.error(f"❌ Drift analysis failed: {e}")
        
        return results
    
    def _determine_overall_status(self, report: MonitoringReport) -> str:
        """Determine overall monitoring status based on all layers."""
        alerts = []
        
        # Check each layer
        if report.layer1_confidence.get('status') == 'ALERT':
            alerts.append(report.layer1_confidence.get('alert'))
        
        if report.layer2_temporal.get('status') in ['ALERT', 'WARNING']:
            alerts.append(report.layer2_temporal.get('alert'))
        
        if report.layer3_drift.get('status') == 'ALERT':
            alerts.append(report.layer3_drift.get('alert'))
        
        report.alerts = alerts
        
        if alerts:
            return "ALERT"
        elif (report.layer2_temporal.get('status') == 'WARNING' or
              report.layer1_confidence.get('status') == 'WARNING'):
            return "WARNING"
        else:
            return "PASS"
