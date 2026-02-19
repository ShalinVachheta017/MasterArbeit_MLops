"""
TENT: Test-Time Entropy Minimization for domain adaptation.

Adapts only the BatchNormalization affine parameters (gamma / beta) by
minimising the entropy of the model's predictions on unlabelled target data.
All other weights (conv kernels, LSTM cells, dense layers) are frozen.

Reference:
    Wang, D., et al. (2021).
    "Tent: Fully Test-time Adaptation by Entropy Minimization."
    ICLR 2021. arXiv:2006.10726

Usage::

    from src.domain_adaptation.tent import tent_adapt

    adapted_model = tent_adapt(
        model,
        target_X,
        n_steps=10,
        learning_rate=1e-4,
        batch_size=64,
    )

Safety rule (built-in):
    If the target distribution is very far OOD (mean entropy > ``ood_entropy_threshold``),
    TENT is skipped and the original model is returned unchanged.
    This prevents catastrophic forgetting when the shift is too extreme.
"""

import copy
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    tf = None
    keras = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def tent_adapt(
    model,
    target_X: np.ndarray,
    n_steps: int = 10,
    learning_rate: float = 1e-4,
    batch_size: int = 64,
    ood_entropy_threshold: float = 0.85,
    copy_model: bool = True,
) -> "keras.Model":
    """
    Adapt BN affine parameters by minimising prediction entropy.

    Parameters
    ----------
    model : keras.Model
        Pre-trained (possibly AdaBN-adapted) source model.
    target_X : np.ndarray, shape (N, window_size, n_channels)
        Unlabelled target-domain data.
    n_steps : int
        Number of gradient-update steps (each processes one mini-batch).
    learning_rate : float
        Learning rate for Adam used only on BN affine params.
    batch_size : int
        Mini-batch size per step.
    ood_entropy_threshold : float
        If the *initial* mean normalised entropy exceeds this value the
        target is considered extreme-OOD and adaptation is skipped.
    copy_model : bool
        If True, work on a deep copy so the original is not modified.

    Returns
    -------
    keras.Model
        Model with updated BN gamma/beta (or original if OOD guard triggered).
    """
    if tf is None:
        raise ImportError("TensorFlow is required for TENT.")

    # ── Safety: OOD guard ──────────────────────────────────────────────
    init_probs = model.predict(target_X[:min(256, len(target_X))], verbose=0)
    eps = 1e-9
    init_entropy = -(init_probs * np.log(init_probs + eps)).sum(axis=1)
    n_classes = init_probs.shape[1]
    norm_entropy = init_entropy / np.log(n_classes)
    mean_norm_entropy = float(norm_entropy.mean())

    if mean_norm_entropy > ood_entropy_threshold:
        logger.warning(
            "TENT skipped — initial mean normalised entropy %.3f > %.3f "
            "(target too far OOD; quarantine recommended).",
            mean_norm_entropy, ood_entropy_threshold,
        )
        return model

    logger.info(
        "TENT: initial mean normalised entropy=%.3f (threshold=%.3f). Adapting…",
        mean_norm_entropy, ood_entropy_threshold,
    )

    # ── Deep copy so original is untouched ────────────────────────────
    if copy_model:
        model = copy.deepcopy(model)

    # ── Freeze everything; unfreeze only BN gamma + beta ──────────────
    bn_param_vars = []
    for layer in model.layers:
        layer.trainable = False
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = True   # allows gradient flow to gamma/beta
            # gamma and beta are weights[0] and weights[1]
            bn_param_vars.extend([layer.gamma, layer.beta])

    if not bn_param_vars:
        logger.warning("No BN layers found — TENT is a no-op.")
        return model

    logger.info("TENT: optimising %d BN affine tensors over %d steps.", len(bn_param_vars), n_steps)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    n_samples  = len(target_X)

    for step in range(n_steps):
        idx   = np.random.choice(n_samples, size=min(batch_size, n_samples), replace=False)
        batch = tf.constant(target_X[idx].astype(np.float32))

        with tf.GradientTape() as tape:
            probs  = model(batch, training=True)   # BN in train mode updates running stats too
            # Entropy: H(p) = -sum(p * log p)  — minimise mean entropy
            ent    = -tf.reduce_sum(probs * tf.math.log(probs + eps), axis=-1)
            loss   = tf.reduce_mean(ent)

        grads = tape.gradient(loss, bn_param_vars)
        optimizer.apply_gradients(zip(grads, bn_param_vars))

        if (step + 1) % max(1, n_steps // 5) == 0:
            logger.debug("  TENT step %d/%d — entropy_loss=%.4f", step + 1, n_steps, float(loss))

    # Final entropy after adaptation
    final_probs      = model.predict(target_X[:min(256, len(target_X))], verbose=0)
    final_entropy    = -(final_probs * np.log(final_probs + eps)).sum(axis=1)
    final_norm_ent   = float((final_entropy / np.log(n_classes)).mean())
    logger.info(
        "TENT done — entropy: %.3f → %.3f (Δ=%.3f)",
        mean_norm_entropy, final_norm_ent, final_norm_ent - mean_norm_entropy,
    )

    return model


def tent_score(model, target_X: np.ndarray, batch_size: int = 64) -> dict:
    """
    Compute entropy-based quality metrics on target data after adaptation.
    Useful for comparing before/after TENT in MLflow logs.
    """
    probs = model.predict(target_X, batch_size=batch_size, verbose=0)
    n_classes = probs.shape[1]
    eps         = 1e-9
    entropy     = -(probs * np.log(probs + eps)).sum(axis=1)
    norm_entropy = entropy / np.log(n_classes)
    confidence   = probs.max(axis=1)

    return {
        "mean_normalised_entropy": float(norm_entropy.mean()),
        "mean_confidence":         float(confidence.mean()),
        "low_confidence_ratio":    float((confidence < 0.5).mean()),
        "n_samples":               int(len(probs)),
    }
