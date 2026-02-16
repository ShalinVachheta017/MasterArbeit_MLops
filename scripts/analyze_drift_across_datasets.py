"""
Analyze drift across all raw datasets vs training baseline.
Produces data-driven threshold recommendation.
"""
import numpy as np
import json
import os
import glob
import pandas as pd

channels = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']

# Load normalized baseline
with open('models/normalized_baseline.json') as f:
    nb = json.load(f)
b_mean = np.array(nb['mean'])
b_std = np.array(nb['std'])

# Load scaler
with open('data/prepared/config.json') as f:
    config = json.load(f)
scaler_mean = np.array(config['scaler_mean'])
scaler_scale = np.array(config['scaler_scale'])

# Find all raw CSV/XLSX accelerometer files
raw_dir = 'data/raw'
accel_files = sorted(glob.glob(os.path.join(raw_dir, '*accelerometer*')))
print(f"Found {len(accel_files)} accelerometer files")

drift_results = []
for accel_f in accel_files:
    base = os.path.basename(accel_f)
    pair_id = base.replace('_accelerometer.csv', '').replace('-accelerometer_data.xlsx', '')
    gyro_f = accel_f.replace('accelerometer', 'gyroscope')
    if not os.path.exists(gyro_f):
        continue
    try:
        if accel_f.endswith('.csv'):
            a = pd.read_csv(accel_f)
            g = pd.read_csv(gyro_f)
        else:
            a = pd.read_excel(accel_f)
            g = pd.read_excel(gyro_f)
        if len(a) < 100 or len(g) < 100:
            continue
        # Get sensor columns (handle various naming conventions)
        accel_cols = [c for c in a.columns if any(k in c.lower() for k in ['accel_x', 'accel_y', 'accel_z'])]
        if not accel_cols:
            accel_cols = [c for c in a.columns if c in ['Ax', 'Ay', 'Az', 'x', 'y', 'z']]
        gyro_cols = [c for c in g.columns if any(k in c.lower() for k in ['gyro_x', 'gyro_y', 'gyro_z'])]
        if not gyro_cols:
            gyro_cols = [c for c in g.columns if c in ['Gx', 'Gy', 'Gz', 'x', 'y', 'z']]
        if len(accel_cols) < 3 or len(gyro_cols) < 3:
            continue
        a_vals = a[accel_cols[:3]].values.astype(float)
        g_vals = g[gyro_cols[:3]].values.astype(float)
        min_len = min(len(a_vals), len(g_vals))
        data_6ch = np.hstack([a_vals[:min_len], g_vals[:min_len]])
        
        # Check if units are milliG (accel > 20 means milliG)
        accel_max = np.abs(data_6ch[:, :3]).max()
        if accel_max > 20:
            data_6ch[:, :3] *= 0.00981  # milliG -> m/s²
        
        # Normalize with training scaler
        data_norm = (data_6ch - scaler_mean) / scaler_scale
        d_mean = data_norm.mean(axis=0)
        
        # Drift vs baseline
        drift = np.abs(d_mean - b_mean) / (b_std + 1e-8)
        drift_results.append({
            'file': pair_id[:30],
            'rows': min_len,
            'max_drift': float(drift.max()),
            'mean_drift': float(drift.mean()),
            'worst_ch': channels[drift.argmax()],
            'per_ch': drift.tolist(),
        })
    except Exception as e:
        print(f"  Skip {base}: {e}")

print(f"\nAnalyzed {len(drift_results)} datasets\n")
header = f"{'File':<32} {'Rows':>8} {'MaxDrift':>9} {'MeanDrift':>10} {'WorstCh':>8}"
print(header)
print("-" * len(header))
drifts = []
for r in sorted(drift_results, key=lambda x: x['max_drift']):
    print(f"{r['file']:<32} {r['rows']:>8} {r['max_drift']:>9.4f} {r['mean_drift']:>10.4f} {r['worst_ch']:>8}")
    drifts.append(r['max_drift'])

drifts = np.array(drifts)
print(f"\n{'='*60}")
print(f"DRIFT DISTRIBUTION ACROSS {len(drifts)} DATASETS")
print(f"{'='*60}")
print(f"  Min:    {drifts.min():.4f}")
print(f"  25th:   {np.percentile(drifts, 25):.4f}")
print(f"  Median: {np.median(drifts):.4f}")
print(f"  75th:   {np.percentile(drifts, 75):.4f}")
print(f"  90th:   {np.percentile(drifts, 90):.4f}")
print(f"  95th:   {np.percentile(drifts, 95):.4f}")
print(f"  Max:    {drifts.max():.4f}")
print(f"  Mean +/- Std: {drifts.mean():.4f} +/- {drifts.std():.4f}")
print()
print(f"  RECOMMENDATIONS:")
print(f"  - Median-based threshold: {np.median(drifts):.4f}")
print(f"  - 75th percentile:        {np.percentile(drifts, 75):.4f}")
print(f"  - Mean + 1*std:           {drifts.mean() + drifts.std():.4f}")
print(f"  - Mean + 2*std:           {drifts.mean() + 2*drifts.std():.4f}")

# Also show per-channel drift stats
print(f"\n{'='*60}")
print(f"PER-CHANNEL DRIFT STATISTICS")
print(f"{'='*60}")
all_ch_drifts = np.array([r['per_ch'] for r in drift_results])
for i, ch in enumerate(channels):
    ch_d = all_ch_drifts[:, i]
    print(f"  {ch}: mean={ch_d.mean():.4f}, std={ch_d.std():.4f}, "
          f"median={np.median(ch_d):.4f}, max={ch_d.max():.4f}")
