"""
Data Validator Module
=====================
Production-grade data validation for sensor data preprocessing.

This module provides validation checks to catch bad data early in the pipeline,
preventing garbage-in-garbage-out scenarios.

Usage:
    from data_validator import DataValidator, ValidationResult
    
    validator = DataValidator()
    result = validator.validate(df)
    
    if not result.is_valid:
        raise ValueError(f"Data validation failed: {result.errors}")
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml


@dataclass
class ValidationResult:
    """
    Result of data validation.
    
    Attributes:
        is_valid: True if data passed all validation checks
        errors: List of critical errors that must be fixed
        warnings: List of non-critical issues to be aware of
        stats: Dictionary of computed statistics
    """
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    
    def __str__(self) -> str:
        status = "✅ VALID" if self.is_valid else "❌ INVALID"
        return f"ValidationResult({status}, errors={len(self.errors)}, warnings={len(self.warnings)})"


class DataValidator:
    """
    Production-grade data validation for sensor data.
    
    Validates sensor data against configurable thresholds for:
    - Missing values
    - Value ranges (accelerometer, gyroscope)
    - Sampling rate consistency
    - Data types
    - Required columns
    
    Example:
        >>> validator = DataValidator()
        >>> result = validator.validate(df)
        >>> if result.is_valid:
        ...     print("Data is ready for processing")
        ... else:
        ...     print(f"Errors: {result.errors}")
    """
    
    DEFAULT_SENSOR_COLUMNS = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    
    def __init__(
        self,
        sensor_columns: Optional[List[str]] = None,
        expected_frequency_hz: float = 50.0,
        max_acceleration: float = 50.0,
        max_gyroscope: float = 500.0,
        max_missing_ratio: float = 0.05,
        config_path: Optional[Path] = None,
    ):
        """
        Initialize DataValidator with validation parameters.
        
        Args:
            sensor_columns: List of sensor column names to validate.
                            Defaults to ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
            expected_frequency_hz: Expected sampling frequency in Hz
            max_acceleration: Maximum reasonable acceleration in m/s²
            max_gyroscope: Maximum reasonable gyroscope in deg/s
            max_missing_ratio: Maximum allowed ratio of missing values (0.05 = 5%)
            config_path: Optional path to YAML config file to load settings from
        """
        # Load from config if provided
        if config_path and config_path.exists():
            self._load_config(config_path)
        else:
            self.sensor_columns = sensor_columns or self.DEFAULT_SENSOR_COLUMNS
            self.expected_frequency_hz = expected_frequency_hz
            self.max_acceleration = max_acceleration
            self.max_gyroscope = max_gyroscope
            self.max_missing_ratio = max_missing_ratio
        
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self, config_path: Path) -> None:
        """Load validation settings from YAML config file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        validation_config = config.get('validation', {})
        thresholds = validation_config.get('thresholds', {})
        
        self.sensor_columns = self.DEFAULT_SENSOR_COLUMNS
        self.expected_frequency_hz = config.get('preprocessing', {}).get('sampling_frequency_hz', 50.0)
        self.max_missing_ratio = thresholds.get('max_missing_ratio', 0.05)
        self.max_acceleration = thresholds.get('max_acceleration_ms2', 50.0)
        self.max_gyroscope = thresholds.get('max_gyroscope_dps', 500.0)
    
    def validate(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp'
    ) -> ValidationResult:
        """
        Validate sensor data DataFrame.
        
        Performs the following checks:
        1. Required columns exist
        2. Data types are numeric
        3. Missing value ratio within threshold
        4. Value ranges are reasonable
        5. Sampling rate is consistent (if timestamp available)
        
        Args:
            df: Input DataFrame with sensor data
            timestamp_col: Name of timestamp column (optional check)
        
        Returns:
            ValidationResult with is_valid flag, errors, warnings, and stats
        """
        errors: List[str] = []
        warnings: List[str] = []
        stats: dict = {}
        
        # 1. Check required columns exist
        missing_cols = [c for c in self.sensor_columns if c not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # 2. Check data types
        for col in self.sensor_columns:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    errors.append(f"Column '{col}' is not numeric: {df[col].dtype}")
        
        # 3. Check missing values
        for col in self.sensor_columns:
            if col in df.columns:
                missing_ratio = df[col].isnull().sum() / len(df)
                stats[f'{col}_missing_ratio'] = missing_ratio
                
                if missing_ratio > self.max_missing_ratio:
                    errors.append(
                        f"Column '{col}' has {missing_ratio:.1%} missing values "
                        f"(max: {self.max_missing_ratio:.1%})"
                    )
                elif missing_ratio > 0:
                    warnings.append(f"Column '{col}' has {missing_ratio:.1%} missing values")
        
        # 4. Check value ranges (accelerometer)
        for col in ['Ax', 'Ay', 'Az']:
            if col in df.columns:
                col_max = df[col].abs().max()
                stats[f'{col}_max_abs'] = col_max
                
                if col_max > self.max_acceleration:
                    warnings.append(
                        f"Column '{col}' has extreme values (max: {col_max:.2f} m/s²)"
                    )
        
        # 5. Check value ranges (gyroscope)
        for col in ['Gx', 'Gy', 'Gz']:
            if col in df.columns:
                col_max = df[col].abs().max()
                stats[f'{col}_max_abs'] = col_max
                
                if col_max > self.max_gyroscope:
                    warnings.append(
                        f"Column '{col}' has extreme values (max: {col_max:.2f} deg/s)"
                    )
        
        # 6. Check sampling rate (if timestamp available)
        if timestamp_col in df.columns and len(df) > 1:
            try:
                timestamps = pd.to_datetime(df[timestamp_col])
                time_diffs = timestamps.diff().dropna()
                mean_period = time_diffs.mean().total_seconds()
                actual_freq = 1.0 / mean_period if mean_period > 0 else 0
                stats['actual_frequency_hz'] = actual_freq
                
                freq_deviation = abs(actual_freq - self.expected_frequency_hz) / self.expected_frequency_hz
                if freq_deviation > 0.1:  # 10% tolerance
                    warnings.append(
                        f"Sampling frequency is {actual_freq:.1f} Hz "
                        f"(expected: {self.expected_frequency_hz} Hz)"
                    )
            except Exception as e:
                warnings.append(f"Could not verify sampling rate: {e}")
        
        # 7. Compute basic statistics
        for col in self.sensor_columns:
            if col in df.columns:
                stats[f'{col}_mean'] = df[col].mean()
                stats[f'{col}_std'] = df[col].std()
                stats[f'{col}_min'] = df[col].min()
                stats[f'{col}_max'] = df[col].max()
        
        is_valid = len(errors) == 0
        
        result = ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            stats=stats
        )
        
        # Log results
        if is_valid:
            self.logger.info(f"✅ Data validation passed ({len(warnings)} warnings)")
        else:
            self.logger.error(f"❌ Data validation failed: {errors}")
        
        return result
    
    def validate_and_raise(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp'
    ) -> ValidationResult:
        """
        Validate data and raise exception if invalid.
        
        Convenience method for pipeline integration where invalid data
        should halt processing.
        
        Args:
            df: Input DataFrame with sensor data
            timestamp_col: Name of timestamp column
        
        Returns:
            ValidationResult if valid
        
        Raises:
            ValueError: If data validation fails
        """
        result = self.validate(df, timestamp_col)
        
        if not result.is_valid:
            raise ValueError(
                f"Data validation failed with {len(result.errors)} errors: "
                f"{'; '.join(result.errors)}"
            )
        
        return result


def load_validator_from_config(config_path: Path) -> DataValidator:
    """
    Factory function to create DataValidator from config file.
    
    Args:
        config_path: Path to pipeline_config.yaml
    
    Returns:
        Configured DataValidator instance
    """
    return DataValidator(config_path=config_path)


# ============================================================================
# Command-line interface for testing
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_validator.py <csv_file> [config_file]")
        sys.exit(1)
    
    csv_file = Path(sys.argv[1])
    config_file = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    # Load data
    print(f"Loading {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Create validator
    if config_file:
        validator = load_validator_from_config(config_file)
    else:
        validator = DataValidator()
    
    # Validate
    result = validator.validate(df)
    
    # Print results
    print(f"\n{result}")
    print(f"\nErrors ({len(result.errors)}):")
    for e in result.errors:
        print(f"  ❌ {e}")
    
    print(f"\nWarnings ({len(result.warnings)}):")
    for w in result.warnings:
        print(f"  ⚠️ {w}")
    
    print(f"\nStatistics (sample):")
    for key in ['Az_mean', 'Az_std', 'Az_max_abs']:
        if key in result.stats:
            print(f"  {key}: {result.stats[key]:.4f}")
    
    sys.exit(0 if result.is_valid else 1)
