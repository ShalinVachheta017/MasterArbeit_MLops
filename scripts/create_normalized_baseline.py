#!/usr/bin/env python3
"""
Create normalized baseline from training data.

This script creates a proper baseline for drift detection by:
1. Loading raw training data
2. Applying the same normalization as production pipeline
3. Storing statistics AND real samples for KS tests
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

def main():
    # Paths
    training_csv = Path('D:/study apply/ML Ops/MasterArbeit_MLops/data/raw/all_users_data_labeled.csv')
    scaler_config = Path('D:/study apply/ML Ops/MasterArbeit_MLops/models/archived_experiments/cv_training_20260106/scaler_config.json')
    output_path = Path('D:/study apply/ML Ops/MasterArbeit_MLops/models/normalized_baseline.json')
    
    # Load scaler parameters
    with open(scaler_config) as f:
        scaler = json.load(f)
    
    # Read raw data
    df = pd.read_csv(training_csv)
    print(f'Loaded training data: {len(df)} rows')
    
    # Get sensor columns (with _w suffix)
    sensor_cols = ['Ax_w', 'Ay_w', 'Az_w', 'Gx_w', 'Gy_w', 'Gz_w']
    channels = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    
    # Apply normalization: (x - mean) / scale
    raw_data = df[sensor_cols].values
    normalized_data = (raw_data - np.array(scaler['mean'])) / np.array(scaler['scale'])
    
    print(f'Normalized data shape: {normalized_data.shape}')
    
    # Create baseline with real normalized samples
    baseline = {
        'created_at': datetime.now().isoformat(),
        'source': 'Normalized training data (StandardScaler applied)',
        'description': 'Statistics from normalized windowed training data',
        'n_samples': len(normalized_data),
        'preprocessing': {
            'scaler_mean': scaler['mean'],
            'scaler_scale': scaler['scale'],
        },
        'per_channel': {}
    }
    
    # Store stats and real samples for each channel
    n_samples_to_store = 10000  # Store 10k samples per channel
    rng = np.random.RandomState(42)
    
    for ch_idx, ch in enumerate(channels):
        ch_data = normalized_data[:, ch_idx]
        
        # Sample indices for stored samples
        sample_indices = rng.choice(len(ch_data), min(n_samples_to_store, len(ch_data)), replace=False)
        samples = ch_data[sample_indices].tolist()
        
        # Histogram for PSI
        hist_counts, hist_edges = np.histogram(ch_data, bins=50)
        
        baseline['per_channel'][ch] = {
            'mean': float(np.mean(ch_data)),
            'std': float(np.std(ch_data)),
            'min': float(np.min(ch_data)),
            'max': float(np.max(ch_data)),
            'percentile_5': float(np.percentile(ch_data, 5)),
            'percentile_50': float(np.percentile(ch_data, 50)),
            'percentile_95': float(np.percentile(ch_data, 95)),
            'samples': samples,  # Real samples for KS test
            'histogram': {
                'counts': hist_counts.tolist(),
                'bin_edges': hist_edges.tolist()
            }
        }
        
        ch_stats = baseline['per_channel'][ch]
        print(f'{ch}: mean={ch_stats["mean"]:.4f}, std={ch_stats["std"]:.4f}')
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(baseline, f, indent=2)
    
    print()
    print(f'Saved normalized baseline with real samples to: {output_path}')
    print(f'File size: {output_path.stat().st_size / 1024:.1f} KB')

if __name__ == '__main__':
    main()
