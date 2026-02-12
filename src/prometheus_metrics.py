#!/usr/bin/env python3
"""
Prometheus Metrics Configuration and Exporter
==============================================

Defines and exports Prometheus metrics for the HAR MLOps pipeline.
Enables Grafana dashboards for real-time monitoring.

Metrics exported:
- Model performance (F1, accuracy, predictions)
- Data drift indicators (PSI, KS-stat)
- Proxy metrics (confidence, entropy, flip rate)
- System metrics (latency, throughput)
- Trigger state

Usage:
    from prometheus_metrics import MetricsExporter, start_metrics_server
    
    exporter = MetricsExporter()
    exporter.record_prediction(confidence=0.95, latency_ms=50)
    
    # Start HTTP server for Prometheus scraping
    start_metrics_server(port=8000)

Author: HAR MLOps Pipeline
Date: January 30, 2026
"""

import time
import threading
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from http.server import HTTPServer, BaseHTTPRequestHandler
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# METRIC DEFINITIONS (Prometheus-compatible format)
# ============================================================================

@dataclass
class MetricDefinition:
    """Definition of a Prometheus metric."""
    name: str
    type: str  # 'counter', 'gauge', 'histogram', 'summary'
    help: str
    labels: List[str] = None
    buckets: List[float] = None  # For histograms


# HAR Pipeline metrics definitions
METRIC_DEFINITIONS = {
    # Model Performance Metrics
    'har_model_f1_score': MetricDefinition(
        name='har_model_f1_score',
        type='gauge',
        help='Current model F1 score (macro-averaged)',
        labels=['model_version', 'fold']
    ),
    'har_model_accuracy': MetricDefinition(
        name='har_model_accuracy',
        type='gauge',
        help='Current model accuracy',
        labels=['model_version']
    ),
    'har_predictions_total': MetricDefinition(
        name='har_predictions_total',
        type='counter',
        help='Total number of predictions made',
        labels=['activity_class']
    ),
    
    # Drift Detection Metrics
    'har_drift_psi': MetricDefinition(
        name='har_drift_psi',
        type='gauge',
        help='Population Stability Index for feature drift',
        labels=['feature']
    ),
    'har_drift_ks_stat': MetricDefinition(
        name='har_drift_ks_stat',
        type='gauge',
        help='Kolmogorov-Smirnov statistic for distribution drift',
        labels=['feature']
    ),
    'har_drift_detected': MetricDefinition(
        name='har_drift_detected',
        type='gauge',
        help='Binary indicator if drift is detected (1=yes, 0=no)',
        labels=['drift_type']
    ),
    
    # Proxy Metrics (Confidence-based monitoring)
    'har_confidence_mean': MetricDefinition(
        name='har_confidence_mean',
        type='gauge',
        help='Mean prediction confidence in current window'
    ),
    'har_confidence_histogram': MetricDefinition(
        name='har_confidence_histogram',
        type='histogram',
        help='Distribution of prediction confidences',
        buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    ),
    'har_entropy_mean': MetricDefinition(
        name='har_entropy_mean',
        type='gauge',
        help='Mean prediction entropy in current window'
    ),
    'har_flip_rate': MetricDefinition(
        name='har_flip_rate',
        type='gauge',
        help='Rate of prediction flips in sliding window'
    ),
    
    # OOD Detection Metrics
    'har_ood_ratio': MetricDefinition(
        name='har_ood_ratio',
        type='gauge',
        help='Ratio of out-of-distribution samples detected'
    ),
    'har_energy_score_mean': MetricDefinition(
        name='har_energy_score_mean',
        type='gauge',
        help='Mean energy score (OOD indicator)'
    ),
    
    # Trigger State Metrics
    'har_trigger_state': MetricDefinition(
        name='har_trigger_state',
        type='gauge',
        help='Current trigger state (0=normal, 1=warning, 2=triggered)',
        labels=['trigger_type']
    ),
    'har_consecutive_warnings': MetricDefinition(
        name='har_consecutive_warnings',
        type='gauge',
        help='Number of consecutive warning states'
    ),
    'har_retraining_triggered_total': MetricDefinition(
        name='har_retraining_triggered_total',
        type='counter',
        help='Total number of retraining triggers'
    ),
    
    # System Metrics
    'har_inference_latency_seconds': MetricDefinition(
        name='har_inference_latency_seconds',
        type='histogram',
        help='Inference latency in seconds',
        buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
    ),
    'har_batch_processing_seconds': MetricDefinition(
        name='har_batch_processing_seconds',
        type='histogram',
        help='Batch processing time in seconds',
        buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0]
    ),
    'har_samples_processed_total': MetricDefinition(
        name='har_samples_processed_total',
        type='counter',
        help='Total samples processed'
    )
}


# ============================================================================
# METRIC STORAGE (Thread-safe)
# ============================================================================

class MetricValue:
    """Thread-safe metric value storage."""
    
    def __init__(self, metric_type: str, buckets: List[float] = None):
        self.metric_type = metric_type
        self._lock = threading.Lock()
        
        if metric_type == 'counter':
            self._value = 0.0
        elif metric_type == 'gauge':
            self._value = 0.0
        elif metric_type == 'histogram':
            self._buckets = buckets or [0.01, 0.1, 0.5, 1.0, 5.0]
            self._bucket_counts = [0] * (len(self._buckets) + 1)  # +1 for +Inf
            self._sum = 0.0
            self._count = 0
        
        self._labels_values: Dict[tuple, Any] = {}
    
    def inc(self, value: float = 1.0, labels: tuple = None):
        """Increment counter."""
        with self._lock:
            if labels:
                key = labels
                if key not in self._labels_values:
                    self._labels_values[key] = 0.0
                self._labels_values[key] += value
            else:
                self._value += value
    
    def set(self, value: float, labels: tuple = None):
        """Set gauge value."""
        with self._lock:
            if labels:
                self._labels_values[labels] = value
            else:
                self._value = value
    
    def observe(self, value: float):
        """Observe histogram value."""
        with self._lock:
            self._sum += value
            self._count += 1
            
            for i, bucket in enumerate(self._buckets):
                if value <= bucket:
                    self._bucket_counts[i] += 1
                    break
            else:
                self._bucket_counts[-1] += 1  # +Inf bucket
    
    def get_prometheus_format(self, metric_def: MetricDefinition) -> str:
        """Export in Prometheus text format."""
        lines = []
        lines.append(f"# HELP {metric_def.name} {metric_def.help}")
        lines.append(f"# TYPE {metric_def.name} {metric_def.type}")
        
        with self._lock:
            if metric_def.type == 'histogram':
                cumulative = 0
                for i, bucket in enumerate(self._buckets):
                    cumulative += self._bucket_counts[i]
                    lines.append(f'{metric_def.name}_bucket{{le="{bucket}"}} {cumulative}')
                cumulative += self._bucket_counts[-1]
                lines.append(f'{metric_def.name}_bucket{{le="+Inf"}} {cumulative}')
                lines.append(f'{metric_def.name}_sum {self._sum}')
                lines.append(f'{metric_def.name}_count {self._count}')
            
            elif self._labels_values:
                for labels, value in self._labels_values.items():
                    label_str = ','.join(f'{k}="{v}"' for k, v in zip(metric_def.labels or [], labels))
                    lines.append(f'{metric_def.name}{{{label_str}}} {value}')
            else:
                lines.append(f'{metric_def.name} {self._value}')
        
        return '\n'.join(lines)


# ============================================================================
# METRICS EXPORTER
# ============================================================================

class MetricsExporter:
    """
    Central metrics exporter for the HAR pipeline.
    
    Records metrics and exports them in Prometheus format.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern for global metrics access."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.logger = logging.getLogger(f"{__name__}.MetricsExporter")
        
        # Initialize all metrics
        self._metrics: Dict[str, MetricValue] = {}
        for name, definition in METRIC_DEFINITIONS.items():
            self._metrics[name] = MetricValue(
                definition.type,
                definition.buckets
            )
    
    # ========================================================================
    # RECORDING METHODS
    # ========================================================================
    
    def record_prediction(
        self,
        activity_class: int,
        confidence: float,
        latency_seconds: float
    ):
        """Record a single prediction event."""
        self._metrics['har_predictions_total'].inc(labels=(str(activity_class),))
        self._metrics['har_confidence_histogram'].observe(confidence)
        self._metrics['har_inference_latency_seconds'].observe(latency_seconds)
        self._metrics['har_samples_processed_total'].inc()
    
    def record_batch_metrics(
        self,
        mean_confidence: float,
        mean_entropy: float,
        flip_rate: float,
        processing_time: float,
        n_samples: int
    ):
        """Record batch-level proxy metrics."""
        self._metrics['har_confidence_mean'].set(mean_confidence)
        self._metrics['har_entropy_mean'].set(mean_entropy)
        self._metrics['har_flip_rate'].set(flip_rate)
        self._metrics['har_batch_processing_seconds'].observe(processing_time)
        self._metrics['har_samples_processed_total'].inc(n_samples)
    
    def record_drift_metrics(
        self,
        psi_values: Dict[str, float],
        ks_values: Dict[str, float],
        drift_detected: bool
    ):
        """Record drift detection metrics."""
        for feature, psi in psi_values.items():
            self._metrics['har_drift_psi'].set(psi, labels=(feature,))
        
        for feature, ks in ks_values.items():
            self._metrics['har_drift_ks_stat'].set(ks, labels=(feature,))
        
        self._metrics['har_drift_detected'].set(
            1.0 if drift_detected else 0.0,
            labels=('overall',)
        )
    
    def record_ood_metrics(
        self,
        ood_ratio: float,
        mean_energy: float
    ):
        """Record OOD detection metrics."""
        self._metrics['har_ood_ratio'].set(ood_ratio)
        self._metrics['har_energy_score_mean'].set(mean_energy)
    
    def record_model_metrics(
        self,
        f1_score: float,
        accuracy: float,
        model_version: str = 'current'
    ):
        """Record model performance metrics."""
        self._metrics['har_model_f1_score'].set(f1_score, labels=(model_version, 'all'))
        self._metrics['har_model_accuracy'].set(accuracy, labels=(model_version,))
    
    def record_trigger_state(
        self,
        state: str,  # 'normal', 'warning', 'triggered'
        consecutive_warnings: int
    ):
        """Record trigger policy state."""
        state_map = {'normal': 0, 'warning': 1, 'triggered': 2}
        self._metrics['har_trigger_state'].set(
            state_map.get(state, 0),
            labels=('overall',)
        )
        self._metrics['har_consecutive_warnings'].set(consecutive_warnings)
        
        if state == 'triggered':
            self._metrics['har_retraining_triggered_total'].inc()
    
    def record_from_monitoring_report(self, report: Dict):
        """
        Record metrics from a full monitoring report.
        
        Args:
            report: Output from post_inference_monitoring.py
        """
        # Proxy metrics
        if 'proxy_metrics' in report:
            pm = report['proxy_metrics']
            self.record_batch_metrics(
                mean_confidence=pm.get('mean_confidence', 0),
                mean_entropy=pm.get('mean_entropy', 0),
                flip_rate=pm.get('flip_rate', 0),
                processing_time=0,
                n_samples=pm.get('n_predictions', 0)
            )
        
        # Drift metrics
        if 'drift_detection' in report:
            dd = report['drift_detection']
            psi_values = dd.get('psi_scores', {})
            ks_values = dd.get('ks_scores', {})
            self.record_drift_metrics(
                psi_values=psi_values,
                ks_values=ks_values,
                drift_detected=dd.get('drift_detected', False)
            )
        
        # OOD metrics
        if 'ood_detection' in report:
            ood = report['ood_detection']
            self.record_ood_metrics(
                ood_ratio=ood.get('ensemble_ood_ratio', 0),
                mean_energy=ood.get('mean_ensemble_score', 0)
            )
        
        # Trigger state
        if 'trigger_evaluation' in report:
            te = report['trigger_evaluation']
            alert_level = te.get('alert_level', 'INFO')
            state = 'triggered' if te.get('should_retrain') else ('warning' if alert_level == 'WARNING' else 'normal')
            self.record_trigger_state(
                state=state,
                consecutive_warnings=te.get('consecutive_warnings', 0)
            )
    
    # ========================================================================
    # EXPORT METHODS
    # ========================================================================
    
    def export_prometheus(self) -> str:
        """Export all metrics in Prometheus text format."""
        lines = []
        
        for name, definition in METRIC_DEFINITIONS.items():
            metric = self._metrics.get(name)
            if metric:
                lines.append(metric.get_prometheus_format(definition))
                lines.append('')  # Empty line between metrics
        
        return '\n'.join(lines)
    
    def export_json(self) -> Dict:
        """Export all metrics as JSON (for dashboards/debugging)."""
        data = {}
        
        for name, metric in self._metrics.items():
            definition = METRIC_DEFINITIONS[name]
            
            with metric._lock:
                if definition.type == 'histogram':
                    data[name] = {
                        'type': 'histogram',
                        'sum': metric._sum,
                        'count': metric._count,
                        'buckets': dict(zip(
                            [str(b) for b in metric._buckets] + ['+Inf'],
                            metric._bucket_counts
                        ))
                    }
                elif metric._labels_values:
                    data[name] = {
                        'type': definition.type,
                        'values': {
                            ','.join(str(l) for l in k): v 
                            for k, v in metric._labels_values.items()
                        }
                    }
                else:
                    data[name] = {
                        'type': definition.type,
                        'value': metric._value
                    }
        
        return data


# ============================================================================
# HTTP SERVER FOR PROMETHEUS SCRAPING
# ============================================================================

class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for Prometheus metrics endpoint."""
    
    exporter = None
    
    def do_GET(self):
        if self.path == '/metrics':
            content = self.exporter.export_prometheus()
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))
        
        elif self.path == '/metrics/json':
            content = json.dumps(self.exporter.export_json(), indent=2)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))
        
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'OK')
        
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def start_metrics_server(port: int = 8000, host: str = '0.0.0.0') -> HTTPServer:
    """
    Start HTTP server for Prometheus scraping.
    
    Args:
        port: Port to listen on
        host: Host to bind to
        
    Returns:
        HTTPServer instance
    """
    MetricsHandler.exporter = MetricsExporter()
    
    server = HTTPServer((host, port), MetricsHandler)
    
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    
    logger.info(f"Metrics server started on http://{host}:{port}/metrics")
    
    return server


# ============================================================================
# CLI
# ============================================================================

def main():
    """Demo and testing CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prometheus Metrics Exporter')
    parser.add_argument('--serve', action='store_true', help='Start metrics server')
    parser.add_argument('--port', type=int, default=8000, help='Server port')
    parser.add_argument('--demo', action='store_true', help='Run demo with sample data')
    parser.add_argument('--export', action='store_true', help='Export current metrics')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    exporter = MetricsExporter()
    
    if args.demo:
        print("Recording sample metrics...")
        
        # Simulate predictions
        import random
        for _ in range(100):
            exporter.record_prediction(
                activity_class=random.randint(0, 10),
                confidence=random.uniform(0.6, 0.99),
                latency_seconds=random.uniform(0.01, 0.1)
            )
        
        # Record batch metrics
        exporter.record_batch_metrics(
            mean_confidence=0.87,
            mean_entropy=0.35,
            flip_rate=0.08,
            processing_time=2.5,
            n_samples=100
        )
        
        # Record drift
        exporter.record_drift_metrics(
            psi_values={'acc_x': 0.05, 'acc_y': 0.12, 'acc_z': 0.08},
            ks_values={'acc_x': 0.04, 'acc_y': 0.09, 'acc_z': 0.06},
            drift_detected=False
        )
        
        # Record model metrics
        exporter.record_model_metrics(
            f1_score=0.89,
            accuracy=0.91,
            model_version='v1.0'
        )
        
        # Record trigger state
        exporter.record_trigger_state(state='normal', consecutive_warnings=0)
        
        print("Sample metrics recorded.\n")
    
    if args.export or args.demo:
        print("=" * 60)
        print("PROMETHEUS FORMAT:")
        print("=" * 60)
        print(exporter.export_prometheus())
    
    if args.serve:
        print(f"\nStarting metrics server on port {args.port}...")
        print(f"Endpoints:")
        print(f"  - http://localhost:{args.port}/metrics (Prometheus)")
        print(f"  - http://localhost:{args.port}/metrics/json (JSON)")
        print(f"  - http://localhost:{args.port}/health (Health check)")
        print("\nPress Ctrl+C to stop.")
        
        server = start_metrics_server(port=args.port)
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            server.shutdown()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
