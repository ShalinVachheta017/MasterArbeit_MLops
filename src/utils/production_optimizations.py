#!/usr/bin/env python3
"""
Production Optimization Utilities
==================================

Performance improvements for the HAR MLOps inference pipeline:
- Model caching (eliminates 2-5s cold-load on repeated calls)
- Vectorized batch prediction (replaces Python loops)
- TF-Lite conversion (optional, requires Flex delegate for BiLSTM)

Usage:
    # Benchmark cached vs uncached model loading
    python -m src.utils.production_optimizations --benchmark models/pretrained/fine_tuned_model_1dcnnbilstm.keras

    # Convert model to TF-Lite (optional)
    python -m src.utils.production_optimizations --convert models/pretrained/fine_tuned_model_1dcnnbilstm.keras
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# Model Cache (Singleton)
# ============================================================================

_MODEL_CACHE = {}


def load_model_cached(model_path: Path):
    """
    Load a Keras model with caching.
    
    Subsequent calls with the same path return the cached model,
    eliminating 2-5s cold-load time during batch processing or API serving.
    """
    key = str(model_path.resolve())
    if key not in _MODEL_CACHE:
        import tensorflow as tf
        logger.info("Loading model (cold): %s", model_path.name)
        _MODEL_CACHE[key] = tf.keras.models.load_model(str(model_path))
    else:
        logger.debug("Using cached model: %s", model_path.name)
    return _MODEL_CACHE[key]


def clear_model_cache():
    """Clear all cached models (e.g., after retraining)."""
    _MODEL_CACHE.clear()
    import tensorflow as tf
    tf.keras.backend.clear_session()
    logger.info("Model cache cleared")


# ============================================================================
# Vectorized Batch Prediction
# ============================================================================

def predict_batch_optimized(
    model,
    X: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Run vectorized batch prediction with optimal batch size.
    
    Replaces Python-loop prediction with a single model.predict() call
    using configurable batch sizes for memory/speed trade-off.
    
    Args:
        model: Loaded Keras model
        X: Input array of shape (n_windows, timesteps, channels)
        batch_size: Batch size for prediction (64 is optimal for most models)
    
    Returns:
        probabilities: np.ndarray of shape (n_windows, n_classes)
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    return model.predict(X, batch_size=batch_size, verbose=0)


# ============================================================================
# TF-Lite Conversion
# ============================================================================

def convert_to_tflite(
    keras_model_path: Path,
    output_path: Optional[Path] = None,
    quantize: bool = True,
    representative_data: Optional[np.ndarray] = None,
) -> Path:
    """
    Convert Keras model to TF-Lite with optional INT8 quantization.
    
    Args:
        keras_model_path: Path to .keras model
        output_path: Output .tflite path (default: same dir, .tflite extension)
        quantize: Apply dynamic-range INT8 quantization
        representative_data: Sample data for full INT8 quantization
        
    Returns:
        Path to the converted .tflite model
    """
    import tensorflow as tf

    keras_model_path = Path(keras_model_path)
    if output_path is None:
        output_path = keras_model_path.with_suffix(".tflite")

    logger.info("Converting %s to TF-Lite...", keras_model_path.name)

    model = tf.keras.models.load_model(str(keras_model_path))
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # BiLSTM requires SELECT_TF_OPS for TensorListReserve
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        logger.info("  Dynamic-range INT8 quantization enabled")

        if representative_data is not None:
            def representative_dataset():
                for i in range(min(200, len(representative_data))):
                    yield [representative_data[i:i+1].astype(np.float32)]
            converter.representative_dataset = representative_dataset
            logger.info("  Full INT8 quantization with representative dataset")

    tflite_model = converter.convert()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    keras_size = keras_model_path.stat().st_size / 1024
    tflite_size = output_path.stat().st_size / 1024
    reduction = (1 - tflite_size / keras_size) * 100

    logger.info("  Keras:   %.1f KB", keras_size)
    logger.info("  TF-Lite: %.1f KB (%.1f%% smaller)", tflite_size, reduction)
    logger.info("  Saved:   %s", output_path)

    del model
    tf.keras.backend.clear_session()

    return output_path


# ============================================================================
# TF-Lite Inference
# ============================================================================

class TFLitePredictor:
    """
    Fast inference using TF-Lite runtime.
    
    Usage:
        predictor = TFLitePredictor("model.tflite")
        probabilities = predictor.predict(X)  # X: (n_windows, 200, 6)
    """

    def __init__(self, model_path: Path):
        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_shape = self.input_details[0]["shape"]  # e.g., (1, 200, 6)
        logger.info(
            "TF-Lite model loaded: input=%s, output=%s",
            self.input_shape, self.output_details[0]["shape"],
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Run inference on batch of windows.
        
        Args:
            X: np.ndarray of shape (n_windows, timesteps, channels)
            
        Returns:
            probabilities: np.ndarray of shape (n_windows, n_classes)
        """
        results = []
        X = X.astype(np.float32)
        
        for i in range(len(X)):
            self.interpreter.set_tensor(
                self.input_details[0]["index"], X[i:i+1]
            )
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]["index"])
            results.append(output[0])
        
        return np.array(results)

    def predict_batch(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Predict in batches (memory-friendly for large datasets)."""
        all_probs = []
        for start in range(0, len(X), batch_size):
            batch = X[start : start + batch_size]
            all_probs.append(self.predict(batch))
        return np.concatenate(all_probs, axis=0)


# ============================================================================
# Benchmarking
# ============================================================================

def benchmark_models(
    keras_path: Path,
    tflite_path: Optional[Path] = None,
    n_samples: int = 100,
) -> dict:
    """
    Benchmark model inference speed and caching performance.
    
    Tests:
    1. Cold vs cached model loading
    2. Batch size impact on inference speed
    3. TF-Lite comparison (if available)
    
    Returns dict with timing results.
    """
    import tensorflow as tf

    # Generate dummy data matching model input
    dummy_data = np.random.randn(n_samples, 200, 6).astype(np.float32)

    # Benchmark cold load
    clear_model_cache()
    t0 = time.perf_counter()
    model = load_model_cached(keras_path)
    cold_load_time = time.perf_counter() - t0

    # Benchmark cached load
    t0 = time.perf_counter()
    model = load_model_cached(keras_path)
    cached_load_time = time.perf_counter() - t0

    # Warm up
    model.predict(dummy_data[:1], verbose=0)

    # Benchmark different batch sizes
    batch_results = {}
    for bs in [1, 16, 32, 64]:
        t0 = time.perf_counter()
        output = predict_batch_optimized(model, dummy_data, batch_size=bs)
        elapsed = time.perf_counter() - t0
        batch_results[f"batch_{bs}_ms"] = elapsed * 1000
        batch_results[f"batch_{bs}_per_sample_ms"] = elapsed * 1000 / n_samples

    results = {
        "n_samples": n_samples,
        "cold_load_ms": cold_load_time * 1000,
        "cached_load_ms": cached_load_time * 1000,
        "load_speedup": cold_load_time / max(cached_load_time, 1e-9),
        **batch_results,
    }

    # Benchmark TF-Lite (if available)
    if tflite_path is None:
        tflite_path = keras_path.with_suffix(".tflite")
    
    if tflite_path and tflite_path.exists():
        try:
            predictor = TFLitePredictor(tflite_path)
            predictor.predict(dummy_data[:1])  # warm up
            t0 = time.perf_counter()
            tflite_output = predictor.predict(dummy_data)
            tflite_time = time.perf_counter() - t0
            results["tflite_total_ms"] = tflite_time * 1000
            results["tflite_per_sample_ms"] = tflite_time * 1000 / n_samples
            max_diff = np.max(np.abs(output - tflite_output))
            results["max_output_diff"] = float(max_diff)
        except RuntimeError as e:
            logger.warning("TF-Lite benchmark skipped (Flex delegate needed): %s", e)
            results["tflite_note"] = "Requires Flex delegate for BiLSTM"

    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Production optimization utilities")
    parser.add_argument("--convert", type=Path, help="Convert Keras model to TF-Lite")
    parser.add_argument("--benchmark", type=Path, help="Benchmark Keras vs TF-Lite")
    parser.add_argument("--quantize", action="store_true", default=True, help="Apply INT8 quantization")
    parser.add_argument("--representative-data", type=Path, help="Path to .npy for full INT8 quant")
    parser.add_argument("--output", type=Path, default=None, help="Output .tflite path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

    if args.convert:
        rep_data = None
        if args.representative_data:
            rep_data = np.load(args.representative_data)
        output = convert_to_tflite(
            args.convert,
            output_path=args.output,
            quantize=args.quantize,
            representative_data=rep_data,
        )
        print(f"\nTF-Lite model saved: {output}")

    if args.benchmark:
        results = benchmark_models(args.benchmark, n_samples=200)
        print("\nBenchmark Results:")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
