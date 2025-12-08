"""
Data Comparison Utility
=======================
Compare training vs production data side-by-side:
- Statistics and distributions
- Scaling parameters
- Window counts
- Data drift detection
- Visual summaries

Scientific Background
---------------------
Data drift occurs when production data differs from training data.
Types of drift:
1. Covariate Shift: Input distribution changes
2. Prior Probability Shift: Class distribution changes
3. Concept Drift: Relationship between input and output changes

Thresholds used in this script:
- Mean drift > 0.1: Significant shift in center of distribution
- Std drift > 0.15: Significant change in data variability
- Per-sensor drift > 0.2: Individual sensor calibration issue

Author: Master Thesis MLOps Project
Date: December 8, 2025
"""

import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))
from config import DATA_PREPARED, PROJECT_ROOT, LOGS_DIR


# ============================================================================
# LOGGING SETUP
# ============================================================================

class ComparisonLogger:
    """
    Centralized logging for data comparison.
    
    Logs are saved to: logs/comparison/comparison_YYYYMMDD_HHMMSS.log
    """
    
    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize logger with file and console handlers."""
        self.log_dir = log_dir or LOGS_DIR / "comparison"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"comparison_{timestamp}.log"
        
        self.logger = logging.getLogger("comparison")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        
        # File handler (DEBUG level)
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        
        # Console handler (INFO level)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"📝 Log file: {self.log_file}")
    
    def get_logger(self) -> logging.Logger:
        return self.logger


class DataComparator:
    """Compare training and production datasets"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.data_prepared = DATA_PREPARED
        
        # Setup logging
        if logger is None:
            log_setup = ComparisonLogger()
            self.logger = log_setup.get_logger()
        else:
            self.logger = logger
    
    def load_training_data(self) -> Dict:
        """Load training dataset statistics"""
        self.logger.info("=" * 60)
        self.logger.info("📥 LOADING TRAINING DATA")
        self.logger.info("=" * 60)
        
        # Load config
        config_path = self.data_prepared / "config.json"
        if not config_path.exists():
            self.logger.error(f"❌ Training config not found: {config_path}")
            raise FileNotFoundError(f"Training config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load arrays
        train_X = np.load(self.data_prepared / "train_X.npy")
        train_y = np.load(self.data_prepared / "train_y.npy")
        val_X = np.load(self.data_prepared / "val_X.npy")
        val_y = np.load(self.data_prepared / "val_y.npy")
        test_X = np.load(self.data_prepared / "test_X.npy")
        test_y = np.load(self.data_prepared / "test_y.npy")
        
        # Combine all
        all_X = np.concatenate([train_X, val_X, test_X])
        all_y = np.concatenate([train_y, val_y, test_y])
        
        self.logger.info(f"✅ Loaded {len(all_X):,} total windows")
        self.logger.info(f"   Train: {len(train_X):,}")
        self.logger.info(f"   Val:   {len(val_X):,}")
        self.logger.info(f"   Test:  {len(test_X):,}")
        
        return {
            'config': config,
            'train_X': train_X,
            'train_y': train_y,
            'val_X': val_X,
            'val_y': val_y,
            'test_X': test_X,
            'test_y': test_y,
            'all_X': all_X,
            'all_y': all_y,
        }
    
    def load_production_data(self) -> Dict:
        """Load production dataset"""
        self.logger.info("=" * 60)
        self.logger.info("📥 LOADING PRODUCTION DATA")
        self.logger.info("=" * 60)
        
        # Load metadata
        meta_path = self.data_prepared / "production_metadata.json"
        if not meta_path.exists():
            self.logger.error(f"❌ Production metadata not found: {meta_path}")
            raise FileNotFoundError(f"Production metadata not found: {meta_path}")
        
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        # Load array
        prod_X = np.load(self.data_prepared / "production_X.npy")
        
        self.logger.info(f"✅ Loaded {len(prod_X):,} windows")
        
        return {
            'metadata': metadata,
            'X': prod_X,
        }
    
    def compare_shapes(self, train_data: Dict, prod_data: Dict):
        """Compare data shapes"""
        self.logger.info("=" * 60)
        self.logger.info("📐 DATA SHAPES COMPARISON")
        self.logger.info("=" * 60)
        
        self.logger.info("Training Data:")
        self.logger.info(f"   Shape: {train_data['all_X'].shape}")
        self.logger.info(f"   Windows: {train_data['all_X'].shape[0]:,}")
        self.logger.info(f"   Window size: {train_data['all_X'].shape[1]}")
        self.logger.info(f"   Sensors: {train_data['all_X'].shape[2]}")
        self.logger.info(f"   Labels: {len(train_data['all_y']):,}")
        self.logger.info(f"   Unique classes: {len(np.unique(train_data['all_y']))}")
        
        self.logger.info("Production Data:")
        self.logger.info(f"   Shape: {prod_data['X'].shape}")
        self.logger.info(f"   Windows: {prod_data['X'].shape[0]:,}")
        self.logger.info(f"   Window size: {prod_data['X'].shape[1]}")
        self.logger.info(f"   Sensors: {prod_data['X'].shape[2]}")
        self.logger.info(f"   Labels: None (unlabeled)")
        
        # Check compatibility
        if train_data['all_X'].shape[1:] == prod_data['X'].shape[1:]:
            self.logger.info("✅ Shape compatibility: MATCH (window_size, sensors)")
        else:
            self.logger.warning("⚠️ Shape compatibility: MISMATCH!")
            self.logger.warning(f"   Training: {train_data['all_X'].shape[1:]}")
            self.logger.warning(f"   Production: {prod_data['X'].shape[1:]}")
    
    def compare_statistics(self, train_data: Dict, prod_data: Dict):
        """Compare statistical properties"""
        self.logger.info("=" * 60)
        self.logger.info("📊 STATISTICAL COMPARISON")
        self.logger.info("=" * 60)
        
        train_X = train_data['all_X']
        prod_X = prod_data['X']
        
        # Overall statistics
        self.logger.info("Overall Statistics:")
        self.logger.info(f"   {'Metric':<10} {'Training':<15} {'Production':<15} {'Difference':<12}")
        self.logger.info("   " + "-" * 55)
        
        metrics = [
            ('Mean', train_X.mean(), prod_X.mean()),
            ('Std', train_X.std(), prod_X.std()),
            ('Min', train_X.min(), prod_X.min()),
            ('Max', train_X.max(), prod_X.max()),
            ('Median', np.median(train_X), np.median(prod_X)),
        ]
        
        for name, train_val, prod_val in metrics:
            diff = abs(train_val - prod_val)
            self.logger.info(f"   {name:<10} {train_val:<15.4f} {prod_val:<15.4f} {diff:<12.4f}")
        
        # Per-sensor statistics
        self.logger.info("\nPer-Sensor Statistics (Mean ± Std):")
        sensor_names = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
        self.logger.info(f"   {'Sensor':<8} {'Training':<22} {'Production':<22} {'Diff':<10}")
        self.logger.info("   " + "-" * 65)
        
        for i, sensor in enumerate(sensor_names):
            train_mean = train_X[:, :, i].mean()
            train_std = train_X[:, :, i].std()
            prod_mean = prod_X[:, :, i].mean()
            prod_std = prod_X[:, :, i].std()
            diff = abs(train_mean - prod_mean)
            
            self.logger.info(f"   {sensor:<8} {train_mean:>8.4f} ± {train_std:<8.4f}  "
                  f"{prod_mean:>8.4f} ± {prod_std:<8.4f}  {diff:>8.4f}")
    
    def compare_distributions(self, train_data: Dict, prod_data: Dict):
        """Compare value distributions using percentiles"""
        self.logger.info("=" * 60)
        self.logger.info("📈 DISTRIBUTION COMPARISON (Percentiles)")
        self.logger.info("=" * 60)
        
        train_X = train_data['all_X'].flatten()
        prod_X = prod_data['X'].flatten()
        
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        
        self.logger.info(f"   {'Percentile':<12} {'Training':<15} {'Production':<15} {'Diff':<12}")
        self.logger.info("   " + "-" * 55)
        
        for p in percentiles:
            train_val = np.percentile(train_X, p)
            prod_val = np.percentile(prod_X, p)
            diff = abs(train_val - prod_val)
            self.logger.info(f"   {p:>3}th{'':<7} {train_val:>12.4f}   {prod_val:>12.4f}   {diff:>10.4f}")
    
    def detect_data_drift(self, train_data: Dict, prod_data: Dict):
        """Detect potential data drift"""
        self.logger.info("=" * 60)
        self.logger.info("🔍 DATA DRIFT DETECTION")
        self.logger.info("=" * 60)
        
        train_X = train_data['all_X']
        prod_X = prod_data['X']
        
        # Mean difference
        mean_diff = abs(train_X.mean() - prod_X.mean())
        mean_threshold = 0.1
        mean_status = "✅ GOOD" if mean_diff < mean_threshold else "⚠️ CHECK"
        
        # Std difference
        std_diff = abs(train_X.std() - prod_X.std())
        std_threshold = 0.15
        std_status = "✅ GOOD" if std_diff < std_threshold else "⚠️ CHECK"
        
        # Range difference
        train_range = train_X.max() - train_X.min()
        prod_range = prod_X.max() - prod_X.min()
        range_diff = abs(train_range - prod_range) / train_range
        range_threshold = 0.5
        range_status = "✅ GOOD" if range_diff < range_threshold else "⚠️ CHECK"
        
        self.logger.info("Drift Indicators:")
        self.logger.info(f"   Mean diff:  {mean_diff:>8.4f} (threshold: {mean_threshold})  {mean_status}")
        self.logger.info(f"   Std diff:   {std_diff:>8.4f} (threshold: {std_threshold})  {std_status}")
        self.logger.info(f"   Range diff: {range_diff:>8.2%} (threshold: {range_threshold:.0%})  {range_status}")
        
        # Per-sensor drift check
        self.logger.info("\nPer-Sensor Drift (Mean):")
        sensor_names = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
        
        drift_detected = False
        for i, sensor in enumerate(sensor_names):
            train_mean = train_X[:, :, i].mean()
            prod_mean = prod_X[:, :, i].mean()
            diff = abs(train_mean - prod_mean)
            status = "✅" if diff < 0.2 else "⚠️"
            
            if diff >= 0.2:
                drift_detected = True
            
            self.logger.info(f"   {sensor}: {diff:>8.4f}  {status}")
        
        # Overall assessment
        self.logger.info("=" * 60)
        if not drift_detected and mean_diff < mean_threshold and std_diff < std_threshold:
            self.logger.info("✅ ASSESSMENT: No significant data drift detected")
            self.logger.info("   Production data distribution is compatible with training data")
            self.logger.info("   Model predictions should be reliable")
        else:
            self.logger.warning("⚠️ ASSESSMENT: Potential data drift detected")
            self.logger.warning("   Production data differs from training distribution")
            self.logger.warning("   Recommendations:")
            self.logger.warning("      1. Review model predictions carefully")
            self.logger.warning("      2. Consider collecting labeled production samples")
            self.logger.warning("      3. Evaluate model performance on production data")
            self.logger.warning("      4. Consider domain adaptation or fine-tuning")
        self.logger.info("=" * 60)
    
    def compare_preprocessing(self, train_data: Dict, prod_data: Dict):
        """Compare preprocessing configurations"""
        self.logger.info("=" * 60)
        self.logger.info("⚙️ PREPROCESSING CONFIGURATION")
        self.logger.info("=" * 60)
        
        train_config = train_data['config']
        prod_meta = prod_data['metadata']
        
        self.logger.info("Training:")
        self.logger.info(f"   Created: {train_config.get('created_date', 'N/A')}")
        self.logger.info(f"   Window size: {train_config.get('window_size', 'N/A')}")
        self.logger.info(f"   Overlap: {train_config.get('overlap', 'N/A')}")
        self.logger.info(f"   Unit conversion: {train_config.get('conversion_applied', 'N/A')}")
        
        self.logger.info("Production:")
        self.logger.info(f"   Created: {prod_meta.get('created_date', 'N/A')}")
        self.logger.info(f"   Window size: {prod_meta.get('window_size', 'N/A')}")
        self.logger.info(f"   Overlap: {prod_meta.get('overlap', 'N/A')}")
        self.logger.info(f"   Unit conversion: {prod_meta.get('conversion_applied', 'N/A')}")
        
        # Check compatibility
        self.logger.info("Configuration Match:")
        window_match = train_config.get('window_size') == prod_meta.get('window_size')
        overlap_match = train_config.get('overlap') == prod_meta.get('overlap')
        
        self.logger.info(f"   Window size: {'✅ MATCH' if window_match else '❌ MISMATCH'}")
        self.logger.info(f"   Overlap: {'✅ MATCH' if overlap_match else '❌ MISMATCH'}")
        
        # Scaler info
        self.logger.info("Scaler (StandardScaler):")
        if 'scaler_mean' in train_config:
            scaler_mean = np.array(train_config['scaler_mean'])
            scaler_scale = np.array(train_config['scaler_scale'])
            self.logger.info(f"   Mean:  {scaler_mean}")
            self.logger.info(f"   Scale: {scaler_scale}")
        
        # Activity labels (training only)
        if 'activity_to_label' in train_config:
            self.logger.info(f"Activity Labels ({len(train_config['activity_to_label'])} classes):")
            for activity, label in sorted(train_config['activity_to_label'].items(), 
                                         key=lambda x: x[1]):
                self.logger.debug(f"   {label:2d}: {activity}")
    
    def generate_summary_report(self, train_data: Dict, prod_data: Dict):
        """Generate summary comparison report"""
        output_path = PROJECT_ROOT / "data" / "prepared" / "DATA_COMPARISON_REPORT.md"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Training vs Production Data Comparison Report\n\n")
            f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            # Shapes
            f.write("## Data Shapes\n\n")
            f.write("| Dataset | Windows | Window Size | Sensors | Labels |\n")
            f.write("|---------|---------|-------------|---------|--------|\n")
            f.write(f"| Training | {train_data['all_X'].shape[0]:,} | "
                   f"{train_data['all_X'].shape[1]} | {train_data['all_X'].shape[2]} | "
                   f"{len(train_data['all_y']):,} |\n")
            f.write(f"| Production | {prod_data['X'].shape[0]:,} | "
                   f"{prod_data['X'].shape[1]} | {prod_data['X'].shape[2]} | None |\n\n")
            
            # Statistics
            f.write("## Statistical Comparison\n\n")
            f.write("| Metric | Training | Production | Difference |\n")
            f.write("|--------|----------|------------|------------|\n")
            
            train_X = train_data['all_X']
            prod_X = prod_data['X']
            
            metrics = [
                ('Mean', train_X.mean(), prod_X.mean()),
                ('Std', train_X.std(), prod_X.std()),
                ('Min', train_X.min(), prod_X.min()),
                ('Max', train_X.max(), prod_X.max()),
            ]
            
            for name, train_val, prod_val in metrics:
                diff = abs(train_val - prod_val)
                f.write(f"| {name} | {train_val:.4f} | {prod_val:.4f} | {diff:.4f} |\n")
            
            # Drift assessment
            f.write("\n## Data Drift Assessment\n\n")
            mean_diff = abs(train_X.mean() - prod_X.mean())
            std_diff = abs(train_X.std() - prod_X.std())
            
            if mean_diff < 0.1 and std_diff < 0.15:
                f.write("✅ **No significant drift detected**\n\n")
                f.write("Production data distribution is compatible with training data. "
                       "Model predictions should be reliable.\n")
            else:
                f.write("⚠️ **Potential data drift detected**\n\n")
                f.write(f"- Mean difference: {mean_diff:.4f}\n")
                f.write(f"- Std difference: {std_diff:.4f}\n\n")
                f.write("**Recommendations:**\n")
                f.write("1. Review model predictions carefully\n")
                f.write("2. Consider collecting labeled production samples\n")
                f.write("3. Evaluate model performance on production data\n")
            
            f.write("\n---\n\n")
            f.write("*Report generated by compare_data.py*\n")
        
        self.logger.info(f"✅ Summary report saved: {output_path}")
    
    def run_comparison(self):
        """Run full comparison"""
        self.logger.info("=" * 60)
        self.logger.info("🔄 DATA COMPARISON: TRAINING vs PRODUCTION")
        self.logger.info("=" * 60)
        
        try:
            # Load data
            train_data = self.load_training_data()
            prod_data = self.load_production_data()
            
            # Run comparisons
            self.compare_shapes(train_data, prod_data)
            self.compare_statistics(train_data, prod_data)
            self.compare_distributions(train_data, prod_data)
            self.compare_preprocessing(train_data, prod_data)
            self.detect_data_drift(train_data, prod_data)
            
            # Generate report
            self.generate_summary_report(train_data, prod_data)
            
            self.logger.info("=" * 60)
            self.logger.info("✅ COMPARISON COMPLETE")
            self.logger.info("=" * 60)
            
        except FileNotFoundError as e:
            self.logger.error(f"❌ ERROR: {e}")
            self.logger.error("   Please run preprocessing on both training and production data first.")
        except Exception as e:
            self.logger.error(f"❌ ERROR: {e}", exc_info=True)
            raise


def main():
    comparator = DataComparator()
    comparator.run_comparison()


if __name__ == "__main__":
    main()
