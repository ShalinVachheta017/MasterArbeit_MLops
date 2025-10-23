"""
Example usage of the modular data preprocessing pipeline.

This script demonstrates how to use the class-based approach for preprocessing
sensor data with custom configurations and different use cases.
"""

from pathlib import Path
from src.modular_data_preprocessing import (
    SensorDataPipeline, 
    ProcessingConfig, 
    SensorDataLoader,
    DataProcessor,
    SensorFusion,
    Resampler,
    DataExporter,
    LoggerSetup
)


def example_basic_usage():
    """Example of basic usage with default configuration."""
    print("=== Basic Usage Example ===")
    
    # Setup paths
    base_dir = Path(__file__).resolve().parent.parent
    accel_path = base_dir / "data" / "2025-03-23-15-23-10-accelerometer_data.xlsx"
    gyro_path = base_dir / "data" / "2025-03-23-15-23-10-gyroscope_data.xlsx"
    
    # Create pipeline with default configuration
    pipeline = SensorDataPipeline(base_dir)
    
    # Process the sensor files
    pipeline.process_sensor_files(accel_path, gyro_path)
    print("Basic processing completed!")


def example_custom_configuration():
    """Example of using custom configuration parameters."""
    print("\n=== Custom Configuration Example ===")
    
    # Create custom configuration
    custom_config = ProcessingConfig(
        target_hz=100,  # Resample to 100Hz instead of 50Hz
        merge_tolerance_ms=2,  # Allow 2ms tolerance for sensor alignment
        interpolation_limit=3,  # Allow more interpolation points
        log_max_bytes=5_000_000,  # Larger log files
        log_backup_count=5
    )
    
    # Setup paths
    base_dir = Path(__file__).resolve().parent.parent
    accel_path = base_dir / "data" / "2025-03-23-15-23-10-accelerometer_data.xlsx"
    gyro_path = base_dir / "data" / "2025-03-23-15-23-10-gyroscope_data.xlsx"
    
    # Create pipeline with custom configuration
    pipeline = SensorDataPipeline(base_dir, config=custom_config)
    
    # Process the sensor files
    pipeline.process_sensor_files(accel_path, gyro_path)
    print("Custom configuration processing completed!")


def example_component_usage():
    """Example of using individual components separately."""
    print("\n=== Individual Component Usage Example ===")
    
    # Setup paths and logging
    base_dir = Path(__file__).resolve().parent.parent
    log_dir = base_dir / "logs" / "preprocessing"
    logger_setup = LoggerSetup(log_dir)
    logger = logger_setup.get_logger()
    
    # Initialize individual components
    data_loader = SensorDataLoader(logger)
    data_processor = DataProcessor(logger)
    # Note: Other components (sensor_fusion, resampler, data_exporter) 
    # would be used in a complete processing pipeline
    
    # Load and process accelerometer data
    accel_path = base_dir / "data" / "2025-03-23-15-23-10-accelerometer_data.xlsx"
    accel_raw = data_loader.load_sensor_data(accel_path)
    accel_raw = data_loader.normalize_column_names(accel_raw, "accelerometer")
    data_loader.validate_columns(accel_raw, "accelerometer")
    accel_raw = data_loader.parse_list_columns(accel_raw)
    accel_raw = data_loader.filter_valid_rows(accel_raw)
    accel_exploded = data_processor.explode_dataframe(accel_raw)
    accel_processed = data_processor.process_sensor_data(accel_exploded, "accelerometer")
    
    print(f"Processed accelerometer data: {accel_processed.shape}")
    print("Individual component processing completed!")


def example_error_handling():
    """Example of error handling and validation."""
    print("\n=== Error Handling Example ===")
    
    try:
        # Setup paths
        base_dir = Path(__file__).resolve().parent.parent
        log_dir = base_dir / "logs" / "preprocessing"
        logger_setup = LoggerSetup(log_dir)
        logger = logger_setup.get_logger()
        
        # Try to load a non-existent file
        data_loader = SensorDataLoader(logger)
        non_existent_path = base_dir / "data" / "non_existent_file.xlsx"
        
        # This should raise an exception
        data_loader.load_sensor_data(non_existent_path)
        
    except FileNotFoundError as e:
        print(f"Expected error caught: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    print("Error handling example completed!")


def example_batch_processing():
    """Example of processing multiple sensor data files."""
    print("\n=== Batch Processing Example ===")
    
    # Setup paths
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    
    # Find all Excel files in the data directory
    excel_files = list(data_dir.glob("*.xlsx"))
    
    if not excel_files:
        print("No Excel files found in data directory")
        return
    
    # Create pipeline
    pipeline = SensorDataPipeline(base_dir)
    
    # Process each file pair (assuming naming convention)
    accel_files = [f for f in excel_files if "accelerometer" in f.name]
    
    for accel_file in accel_files:
        # Find corresponding gyro file
        base_name = accel_file.name.replace("accelerometer", "").replace("_data.xlsx", "")
        gyro_file = data_dir / f"{base_name}gyroscope_data.xlsx"
        
        if gyro_file.exists():
            print(f"Processing pair: {accel_file.name} + {gyro_file.name}")
            try:
                pipeline.process_sensor_files(accel_file, gyro_file)
                print(f"Successfully processed: {accel_file.name}")
            except (FileNotFoundError, ValueError, KeyError) as e:
                print(f"Failed to process {accel_file.name}: {e}")
        else:
            print(f"No matching gyro file for: {accel_file.name}")
    
    print("Batch processing example completed!")


if __name__ == "__main__":
    # Run all examples
    try:
        example_basic_usage()
        example_custom_configuration()
        example_component_usage()
        example_error_handling()
        example_batch_processing()
        
        print("\n=== All Examples Completed Successfully! ===")
        
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"Example execution failed: {e}")
        print("Make sure the data files exist in the correct location.")
