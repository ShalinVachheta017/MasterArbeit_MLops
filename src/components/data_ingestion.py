"""
Component 1 – Data Ingestion
=============================================================================

Self-contained ingestion that handles ALL raw-data scenarios:

  A. Direct CSV input   (--input-csv my_recording.csv)
  B. Raw sensor files    (accelerometer + gyroscope Excel/CSV in data/raw/)
  C. Decoded fallback    (data/raw/Decoded/ when nothing found in raw root)

Smart file discovery:
  1. Search data/raw/ root for accelerometer + gyroscope file pairs
  2. If not found, search data/raw/Decoded/ (or other sub-folders)
  3. Pair files by filename prefix, pick newest pair

Skip-already-processed:
  - Before processing a file pair, checks if data/processed/ already has
    a fused CSV for that pair (based on a manifest file).
  - If yes, skips processing and reuses existing CSV.
  - This means repeated `python run_pipeline.py` calls won't redo work.

Merges logic from:
  - sensor_data_pipeline.py  (Excel/CSV → time-aligned, fused CSV)
  - preprocess_data.py       (unit detection, gravity removal)

Output: data/processed/sensor_fused_50Hz.csv
=============================================================================
"""

import ast
import json
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.entity.config_entity import DataIngestionConfig, PipelineConfig
from src.entity.artifact_entity import DataIngestionArtifact

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
# Processing Config Defaults
# ────────────────────────────────────────────────────────────────────────
DEFAULT_TARGET_HZ = 50
DEFAULT_MERGE_TOLERANCE_MS = 1
DEFAULT_INTERPOLATION_LIMIT = 2

# Column name mappings (Garmin / decoded → standard x, y, z)
ACCEL_RENAME = {
    "calibrated_accel_x": "x", "calibrated_accel_y": "y",
    "calibrated_accel_z": "z",
    "accel_x": "x", "accel_y": "y", "accel_z": "z",
}
GYRO_RENAME = {
    "calibrated_gyro_x": "x", "calibrated_gyro_y": "y",
    "calibrated_gyro_z": "z",
    "gyro_x": "x", "gyro_y": "y", "gyro_z": "z",
}

# Manifest filename for tracking what has been processed
PROCESSED_MANIFEST = "ingestion_manifest.json"


# ============================================================================
# HELPER: Raw File Discovery
# ============================================================================

def discover_sensor_files(
    raw_dir: Path,
    search_subdirs: bool = True,
) -> List[Tuple[Path, Path]]:
    """Find ALL matching accelerometer/gyroscope file pairs.

    Search order:
      1. data/raw/  (root level)
      2. data/raw/Decoded/  (or any sub-folder)

    Returns a list of (accel_path, gyro_path) pairs, newest first.
    """
    pairs: List[Tuple[Path, Path]] = []

    # Collect candidate directories
    search_dirs = [raw_dir]
    if search_subdirs:
        search_dirs += sorted(
            [d for d in raw_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )

    for directory in search_dirs:
        accel_files = sorted(
            [p for p in directory.glob("*") if p.is_file() and "accelerometer" in p.name.lower()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        gyro_files = sorted(
            [p for p in directory.glob("*") if p.is_file() and "gyroscope" in p.name.lower()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        # Pair by prefix (part before "accelerometer"/"gyroscope")
        used_gyros = set()
        for accel in accel_files:
            prefix = accel.name.lower().split("accelerometer")[0]
            for gyro in gyro_files:
                if id(gyro) in used_gyros:
                    continue
                gyro_prefix = gyro.name.lower().split("gyroscope")[0]
                if prefix == gyro_prefix:
                    pairs.append((accel, gyro))
                    used_gyros.add(id(gyro))
                    break

    # Sort newest first
    pairs.sort(key=lambda p: p[0].stat().st_mtime, reverse=True)
    return pairs


def find_latest_sensor_pair(raw_dir: Path) -> Tuple[Path, Path]:
    """Return the single newest accel/gyro pair (backward-compat helper)."""
    pairs = discover_sensor_files(raw_dir)
    if not pairs:
        raise FileNotFoundError(
            f"No accelerometer/gyroscope file pairs found in {raw_dir} "
            f"or its sub-folders (e.g., Decoded/)."
        )
    return pairs[0]


# ============================================================================
# HELPER: Processing Manifest (skip-already-processed)
# ============================================================================

def _load_manifest(processed_dir: Path) -> Dict:
    manifest_path = processed_dir / PROCESSED_MANIFEST
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            return json.load(f)
    return {"processed_pairs": {}}


def _save_manifest(processed_dir: Path, manifest: Dict):
    manifest_path = processed_dir / PROCESSED_MANIFEST
    processed_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def _pair_key(accel_path: Path, gyro_path: Path) -> str:
    """Unique key for an accel/gyro pair."""
    return f"{accel_path.name}|{gyro_path.name}"


# ============================================================================
# HELPER: Sensor Data Loading & Parsing
# ============================================================================

def _load_sensor_file(file_path: Path) -> pd.DataFrame:
    """Load a sensor file (Excel or CSV)."""
    suffix = file_path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(file_path)
    elif suffix == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
    logger.info("  Loaded %s: %d rows × %d cols", file_path.name, len(df), len(df.columns))
    return df


def _validate_sensor_file(
    df: pd.DataFrame, file_path: Path, sensor_type: str
) -> None:
    """
    Validate sensor file has required data.

    Raises ValueError if file is empty or missing required columns.
    """
    if len(df) == 0:
        raise ValueError(
            f"Empty file: {file_path.name} has 0 rows"
        )

    # Check for required time column ('timestamp' is the raw column;
    # 'true_time' is computed later by _create_timestamps)
    df_cols_lower = [col.lower().strip() for col in df.columns]
    if 'timestamp' not in df_cols_lower:
        raise ValueError(
            f"Missing 'timestamp' column in {file_path.name}"
        )


def _normalize_columns(df: pd.DataFrame, sensor_type: str) -> pd.DataFrame:
    """Rename Garmin-specific column names to standard x/y/z."""
    rename_map = ACCEL_RENAME if sensor_type == "accelerometer" else GYRO_RENAME
    return df.rename(columns=rename_map)


def _parse_list_cell(value):
    """Parse a cell value that may contain a JSON/Python list."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
        parts = [v.strip() for v in value.split(",") if v.strip()]
        if len(parts) > 1:
            try:
                return [float(p) for p in parts]
            except ValueError:
                pass
    if isinstance(value, (int, float)):
        return [value]
    return value


def _parse_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parse list-encoded columns (sample_time_offset, x, y, z)."""
    for col in ["sample_time_offset", "x", "y", "z"]:
        if col in df.columns:
            df[col] = df[col].apply(_parse_list_cell)
    return df


def _filter_valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where x, y, z, offset are lists of same length."""
    list_cols = ["sample_time_offset", "x", "y", "z"]
    present = [c for c in list_cols if c in df.columns]
    if not present:
        return df

    def _valid(row):
        lengths = []
        for c in present:
            val = row[c]
            if not isinstance(val, list):
                return False
            lengths.append(len(val))
        return len(set(lengths)) == 1 and lengths[0] > 0

    mask = df.apply(_valid, axis=1)
    n_dropped = (~mask).sum()
    if n_dropped:
        logger.info("  Dropped %d invalid rows (mismatched array lengths)", n_dropped)
    return df[mask].reset_index(drop=True)


def _explode_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Expand list columns into individual rows."""
    list_cols = ["sample_time_offset", "x", "y", "z"]
    present = [c for c in list_cols if c in df.columns]
    if not present:
        return df
    return df.explode(present, ignore_index=True)


def _create_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Create precise timestamps using base_time + offset."""
    if "timestamp" in df.columns:
        df["base_time"] = pd.to_datetime(df["timestamp"], utc=True)
    if "timestamp_ms" in df.columns:
        df["base_time"] = df["base_time"] + pd.to_timedelta(df["timestamp_ms"], unit="ms")
    if "sample_time_offset" in df.columns:
        df["sample_time_offset"] = pd.to_numeric(df["sample_time_offset"], errors="coerce")
        df["true_time"] = df["base_time"] + pd.to_timedelta(df["sample_time_offset"], unit="ms")
    elif "base_time" in df.columns:
        df["true_time"] = df["base_time"]
    return df


def _rename_axes(df: pd.DataFrame, sensor_type: str) -> pd.DataFrame:
    """Rename x/y/z → Ax/Ay/Az or Gx/Gy/Gz."""
    if sensor_type == "accelerometer":
        return df.rename(columns={"x": "Ax", "y": "Ay", "z": "Az"})
    else:
        return df.rename(columns={"x": "Gx", "y": "Gy", "z": "Gz"})


def _merge_and_resample(
    accel_df: pd.DataFrame,
    gyro_df: pd.DataFrame,
    target_hz: int = DEFAULT_TARGET_HZ,
    merge_tolerance_ms: int = DEFAULT_MERGE_TOLERANCE_MS,
) -> pd.DataFrame:
    """Merge accelerometer and gyroscope data, then resample to target Hz."""
    # Ensure sorted by time
    accel_df = accel_df.sort_values("true_time").reset_index(drop=True)
    gyro_df = gyro_df.sort_values("true_time").reset_index(drop=True)

    # Merge on time
    tolerance = pd.Timedelta(milliseconds=merge_tolerance_ms)
    merged = pd.merge_asof(
        accel_df[["true_time", "Ax", "Ay", "Az"]],
        gyro_df[["true_time", "Gx", "Gy", "Gz"]],
        on="true_time",
        tolerance=tolerance,
        direction="nearest",
    )

    n_matched = merged[["Gx", "Gy", "Gz"]].notna().all(axis=1).sum()
    logger.info("  Merge: %d/%d rows matched (%.1f%%)",
                n_matched, len(merged), 100 * n_matched / max(len(merged), 1))

    # Set time index and resample
    merged = merged.set_index("true_time")
    period = f"{int(1000 / target_hz)}ms"  # e.g., "20ms" for 50Hz
    resampled = merged.resample(period).mean().interpolate(
        method="linear", limit=DEFAULT_INTERPOLATION_LIMIT,
    )
    resampled = resampled.dropna().reset_index()

    # Add convenience columns
    resampled["timestamp_ms"] = (
        resampled["true_time"].astype(np.int64) // 10**6
    )
    resampled["timestamp_iso"] = resampled["true_time"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")

    logger.info("  Resampled to %d Hz: %d rows", target_hz, len(resampled))
    return resampled


# ============================================================================
# DATA INGESTION COMPONENT
# ============================================================================

class DataIngestion:
    """Ingest raw sensor files (Excel/CSV pairs or single CSV) into a fused CSV.

    Smart discovery:
      1. Checks data/raw/ for accelerometer + gyroscope files
      2. Falls back to data/raw/Decoded/ (or sub-folders)
      3. Tracks processed files in a manifest — skips already-processed pairs

    Output: data/processed/sensor_fused_50Hz.csv
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        config: DataIngestionConfig,
    ):
        self.pipeline_config = pipeline_config
        self.config = config

    # ================================================================== #
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """Run data ingestion and return the artifact."""
        logger.info("=" * 60)
        logger.info("STAGE 1 — Data Ingestion")
        logger.info("=" * 60)

        output_dir = Path(self.config.output_dir or self.pipeline_config.data_processed_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── Path A: direct CSV input ──────────────────────────────────
        if self.config.input_csv is not None:
            return self._ingest_csv(Path(self.config.input_csv), output_dir)

        # ── Path B: explicit accel/gyro files ─────────────────────────
        if self.config.accel_file and self.config.gyro_file:
            return self._ingest_sensor_pair(
                Path(self.config.accel_file),
                Path(self.config.gyro_file),
                output_dir,
            )

        # ── Path C: auto-discover sensor files ────────────────────────
        return self._ingest_auto_discover(output_dir)

    # ================================================================== #
    # Path A: CSV Input
    # ================================================================== #
    def _ingest_csv(self, csv_path: Path, output_dir: Path) -> DataIngestionArtifact:
        """Copy / validate a user-provided CSV."""
        logger.info("Ingesting CSV: %s", csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        dest = output_dir / "sensor_fused_50Hz.csv"
        if csv_path.resolve() != dest.resolve():
            shutil.copy2(csv_path, dest)
        logger.info("CSV ingested: %d rows × %d cols → %s", len(df), len(df.columns), dest)

        return DataIngestionArtifact(
            fused_csv_path=dest,
            n_rows=len(df),
            n_columns=len(df.columns),
            sampling_hz=self.config.target_hz,
            ingestion_timestamp=datetime.now().isoformat(),
            source_type="csv",
        )

    # ================================================================== #
    # Path B: Explicit sensor pair
    # ================================================================== #
    def _ingest_sensor_pair(
        self,
        accel_path: Path,
        gyro_path: Path,
        output_dir: Path,
    ) -> DataIngestionArtifact:
        """Process one specific accel/gyro pair."""
        logger.info("Processing sensor pair:")
        logger.info("  Accel: %s", accel_path)
        logger.info("  Gyro:  %s", gyro_path)

        if not accel_path.exists():
            raise FileNotFoundError(f"Accelerometer file not found: {accel_path}")
        if not gyro_path.exists():
            raise FileNotFoundError(f"Gyroscope file not found: {gyro_path}")

        fused_df = self._process_pair(accel_path, gyro_path)
        dest = output_dir / "sensor_fused_50Hz.csv"
        fused_df.to_csv(dest, index=False)

        return DataIngestionArtifact(
            fused_csv_path=dest,
            n_rows=len(fused_df),
            n_columns=len(fused_df.columns),
            sampling_hz=self.config.target_hz,
            ingestion_timestamp=datetime.now().isoformat(),
            source_type="excel",
        )

    # ================================================================== #
    # Path C: Auto-discover + skip-already-processed
    # ================================================================== #
    def _ingest_auto_discover(self, output_dir: Path) -> DataIngestionArtifact:
        """Auto-discover sensor files in data/raw/ (then Decoded/), skip processed."""
        raw_dir = self.pipeline_config.data_raw_dir

        logger.info("Auto-discovering sensor files in: %s", raw_dir)
        all_pairs = discover_sensor_files(raw_dir)

        if not all_pairs:
            # Last resort: check if fused CSV already exists
            existing = output_dir / "sensor_fused_50Hz.csv"
            if existing.exists():
                logger.info("No raw files found, but existing fused CSV available: %s", existing)
                df = pd.read_csv(existing)
                return DataIngestionArtifact(
                    fused_csv_path=existing,
                    n_rows=len(df),
                    n_columns=len(df.columns),
                    sampling_hz=self.config.target_hz,
                    ingestion_timestamp=datetime.now().isoformat(),
                    source_type="existing",
                )
            raise FileNotFoundError(
                f"No accelerometer/gyroscope files found in {raw_dir} "
                f"or sub-folders (e.g., Decoded/). Place your raw files there "
                f"or use --input-csv to provide a fused CSV directly."
            )

        logger.info("Found %d sensor pair(s)", len(all_pairs))
        for i, (a, g) in enumerate(all_pairs):
            logger.info("  [%d] %s  +  %s", i + 1, a.name, g.name)

        # Check manifest for already-processed pairs
        manifest = _load_manifest(output_dir)
        new_pairs = []
        for accel, gyro in all_pairs:
            key = _pair_key(accel, gyro)
            if key in manifest["processed_pairs"]:
                logger.info("  SKIP (already processed): %s", key)
            else:
                new_pairs.append((accel, gyro))

        if not new_pairs:
            # ALL files already processed — reuse latest fused CSV
            latest_csv = output_dir / "sensor_fused_50Hz.csv"
            if latest_csv.exists():
                logger.info("All %d pair(s) already processed. Reusing: %s",
                            len(all_pairs), latest_csv)
                df = pd.read_csv(latest_csv)
                return DataIngestionArtifact(
                    fused_csv_path=latest_csv,
                    n_rows=len(df),
                    n_columns=len(df.columns),
                    sampling_hz=self.config.target_hz,
                    ingestion_timestamp=datetime.now().isoformat(),
                    source_type="cached",
                )
            # Manifest says processed but CSV missing → reprocess newest
            new_pairs = [all_pairs[0]]

        # Process new pairs (use the newest unprocessed one for the main fused CSV)
        logger.info("Processing %d new pair(s) (skipped %d already-processed)",
                     len(new_pairs), len(all_pairs) - len(new_pairs))

        # Try processing pairs until one succeeds (skip empty/corrupt files)
        fused_df = None
        processed_pair = None
        
        for accel_path, gyro_path in new_pairs:
            try:
                logger.info("Processing pair:")
                logger.info("  Accel: %s", accel_path)
                logger.info("  Gyro:  %s", gyro_path)
                
                fused_df = self._process_pair(accel_path, gyro_path)
                processed_pair = (accel_path, gyro_path)
                break  # Success — stop trying other pairs
            
            except (ValueError, KeyError, Exception) as e:
                logger.warning("  ⚠ Skipping pair due to error: %s", str(e))
                continue  # Try next pair
        
        if fused_df is None or processed_pair is None:
            raise ValueError(f"All {len(new_pairs)} sensor pairs failed processing (empty files or corrupt data)")
        
        accel_path, gyro_path = processed_pair
        dest = output_dir / "sensor_fused_50Hz.csv"
        fused_df.to_csv(dest, index=False)

        # Mark as processed in manifest
        manifest["processed_pairs"][_pair_key(accel_path, gyro_path)] = {
            "accel": str(accel_path),
            "gyro": str(gyro_path),
            "fused_csv": str(dest),
            "timestamp": datetime.now().isoformat(),
            "n_rows": len(fused_df),
                }
        _save_manifest(output_dir, manifest)

        logger.info("Fused CSV: %d rows × %d cols → %s",
                     len(fused_df), len(fused_df.columns), dest)

        return DataIngestionArtifact(
            fused_csv_path=dest,
            n_rows=len(fused_df),
            n_columns=len(fused_df.columns),
            sampling_hz=self.config.target_hz,
            ingestion_timestamp=datetime.now().isoformat(),
            source_type="auto_discover",
        )

    # ================================================================== #
    # Core Processing — sensor pair → fused DataFrame
    # ================================================================== #
    def _process_pair(self, accel_path: Path, gyro_path: Path) -> pd.DataFrame:
        """Load, parse, merge, and resample one accel/gyro pair."""
        target_hz = self.config.target_hz
        merge_tol = self.config.merge_tolerance_ms

        # Load
        accel_df = _load_sensor_file(accel_path)
        gyro_df = _load_sensor_file(gyro_path)

        # Validate files (check for empty or missing columns)
        _validate_sensor_file(accel_df, accel_path, "accelerometer")
        _validate_sensor_file(gyro_df, gyro_path, "gyroscope")

        # Normalize column names
        accel_df = _normalize_columns(accel_df, "accelerometer")
        gyro_df = _normalize_columns(gyro_df, "gyroscope")

        # Parse list columns (Garmin stores arrays as strings)
        accel_df = _parse_list_columns(accel_df)
        gyro_df = _parse_list_columns(gyro_df)

        # Filter rows with valid arrays
        accel_df = _filter_valid_rows(accel_df)
        gyro_df = _filter_valid_rows(gyro_df)

        # Explode list columns into individual rows
        accel_df = _explode_rows(accel_df)
        gyro_df = _explode_rows(gyro_df)

        # Create timestamps
        accel_df = _create_timestamps(accel_df)
        gyro_df = _create_timestamps(gyro_df)

        # Rename axes: x/y/z → Ax/Ay/Az and Gx/Gy/Gz
        accel_df = _rename_axes(accel_df, "accelerometer")
        gyro_df = _rename_axes(gyro_df, "gyroscope")

        # Convert numeric columns
        for col in ["Ax", "Ay", "Az"]:
            if col in accel_df.columns:
                accel_df[col] = pd.to_numeric(accel_df[col], errors="coerce")
        for col in ["Gx", "Gy", "Gz"]:
            if col in gyro_df.columns:
                gyro_df[col] = pd.to_numeric(gyro_df[col], errors="coerce")

        # Merge and resample
        fused = _merge_and_resample(accel_df, gyro_df, target_hz, merge_tol)

        return fused
