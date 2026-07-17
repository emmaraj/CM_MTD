"""
config_parser.py
-----------------
Configuration loading and dataset ingestion utilities for the CM-MTD
pipeline. Nothing here hardcodes a dataset's shape — `input_dim` and
`num_classes` are always inferred from the loaded arrays, per the
project's dataset-agnostic requirement.
"""

from __future__ import annotations

import os
import random
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import yaml

logger = logging.getLogger("cm_mtd")


# =============================================================================
# YAML loading
# =============================================================================

def load_config(path: str) -> dict:
    """Load the YAML config into a plain nested dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: dict) -> None:
    required_top_level = ["data", "lstm", "network", "reward", "dqn", "ppo", "training"]
    missing = [k for k in required_top_level if k not in cfg]
    if missing:
        raise KeyError(f"config.yaml is missing required section(s): {missing}")


def set_global_seed(seed: int) -> None:
    """Seed python, numpy, and (if importable) tensorflow for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


# =============================================================================
# Dataset container
# =============================================================================

@dataclass
class Dataset:
    """
    Holds the loaded .npy arrays plus dimensions inferred from them.
    `input_dim` and `num_classes` are the two values the rest of the
    pipeline (LSTM, environment) reads instead of ever hardcoding a number.
    """
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    input_dim: int = field(init=False)
    num_classes: int = field(init=False)
    class_names: list = field(default_factory=list)

    def __post_init__(self):
        if self.X_train.ndim != 2:
            raise ValueError(
                f"Expected X_train to be 2D [n_samples, n_features], got shape {self.X_train.shape}"
            )
        self.input_dim = self.X_train.shape[1]
        self.num_classes = int(max(self.y_train.max(), self.y_test.max())) + 1

        if self.X_test.shape[1] != self.input_dim:
            raise ValueError(
                f"X_train has {self.input_dim} features but X_test has {self.X_test.shape[1]}"
            )
        if not self.class_names:
            self.class_names = [f"class_{i}" for i in range(self.num_classes)]
        if len(self.class_names) != self.num_classes:
            logger.warning(
                "config.data.class_names has %d entries but %d classes were "
                "inferred from the data; falling back to generic names "
                "(class_0, class_1, ...). If anything downstream looks up a "
                "class by name (e.g. environment.py's HAM/RM logic expects "
                "'Infiltration' and 'DoS/DDoS'), it will now fail loudly -- "
                "check config.data.x_train_path/y_train_path aren't pointing "
                "at stale .npy files from a different preprocessing run.",
                len(self.class_names), self.num_classes,
            )
            self.class_names = [f"class_{i}" for i in range(self.num_classes)]


def _stratified_head(X: np.ndarray, y: np.ndarray, max_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Deterministically take the first-occurring rows per class, interleaved
    in original order, until max_samples rows are collected. No randomness
    is used — this keeps dataset truncation reproducible and consistent
    with the "no synthetic/random stepping" requirement.
    """
    classes = np.unique(y)
    per_class_budget = max(1, max_samples // len(classes))

    keep_idx = []
    for c in classes:
        class_idx = np.flatnonzero(y == c)[:per_class_budget]
        keep_idx.append(class_idx)
    keep_idx = np.sort(np.concatenate(keep_idx))

    # If under budget (small classes exhausted), top up with the next
    # available rows in original order, still deterministically.
    if len(keep_idx) < min(max_samples, len(y)):
        remaining_needed = min(max_samples, len(y)) - len(keep_idx)
        mask = np.ones(len(y), dtype=bool)
        mask[keep_idx] = False
        fill_idx = np.flatnonzero(mask)[:remaining_needed]
        keep_idx = np.sort(np.concatenate([keep_idx, fill_idx]))

    return X[keep_idx], y[keep_idx]


def load_dataset(cfg: dict) -> Dataset:
    """
    Load X/y train/test .npy arrays as specified in config['data'] and
    return a Dataset with input_dim / num_classes inferred (never
    hardcoded). Deliberately does NOT truncate here: the LSTM's sliding-
    window training (config_parser is dataset-agnostic and doesn't know
    about windowing, but models.py's build_sliding_windows assumes genuine
    row-to-row temporal adjacency) needs the full, originally-ordered
    data. The RL environment's max_samples cap is applied separately, only
    to the row-cycling view it builds for itself in environment.py -- see
    that module's docstring for why sharing one truncated view between
    both consumers was a bug (stratified truncation reorders/subsamples
    rows, which is fine for a classifier but fragments the temporal
    continuity LSTM sequence windows depend on).
    """
    data_cfg = cfg["data"]

    def _load(path_key: str) -> np.ndarray:
        path = data_cfg[path_key]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Dataset file not found at '{path}' (config.data.{path_key}). "
                f"Run the preprocessing notebook first, or point config.yaml "
                f"at your existing .npy files."
            )
        arr = np.load(path)
        logger.info("Loaded %s -> shape %s dtype %s", path, arr.shape, arr.dtype)
        return arr

    X_train = _load("x_train_path").astype(np.float32)
    X_test = _load("x_test_path").astype(np.float32)
    y_train = _load("y_train_path").astype(np.int64).reshape(-1)
    y_test = _load("y_test_path").astype(np.int64).reshape(-1)

    return Dataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        class_names=list(data_cfg.get("class_names", [])),
    )


def build_env_row_cache(dataset: "Dataset", data_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds the (possibly truncated) X/y view the RL environment cycles
    through for its per-node row assignment. This is the ONLY place
    max_samples/truncation_strategy apply -- LSTM training always sees the
    full dataset via `dataset.X_train`/`dataset.y_train` directly. Row
    order/continuity doesn't matter here the way it does for LSTM
    sequence windows, since environment.py already assigns rows to nodes
    via an artificial deterministic round-robin, not genuine per-node
    temporal adjacency.
    """
    X_train, y_train = dataset.X_train, dataset.y_train
    max_samples = data_cfg.get("max_samples")
    if max_samples is not None and len(X_train) > max_samples:
        strategy = data_cfg.get("truncation_strategy", "head")
        if strategy == "stratified_head":
            X_train, y_train = _stratified_head(X_train, y_train, max_samples)
        elif strategy == "head":
            X_train, y_train = X_train[:max_samples], y_train[:max_samples]
        else:
            raise ValueError(f"Unknown truncation_strategy: {strategy}")
        logger.info(
            "Environment row cache truncated to %d rows via '%s' strategy "
            "(bounds memory for the RL loop; LSTM training is unaffected "
            "and uses the full dataset).",
            len(X_train), strategy,
        )
    return X_train, y_train


# =============================================================================
# Class weighting (guards against the LSTM-collapse failure mode seen with
# CICIDS-2017's severely under-represented Infiltration class)
# =============================================================================

def compute_class_weights(y: np.ndarray, num_classes: int, strategy: str = "balanced_capped",
                           max_ratio: float = 20.0) -> Optional[dict]:
    """
    Compute per-class weights for model.fit(..., class_weight=...).

    strategy:
      "none"            -> None (no weighting)
      "balanced"        -> classic inverse-frequency weighting (sklearn-style)
      "balanced_capped" -> inverse-frequency weighting, then clipped so that
                            max_weight <= max_ratio * min_weight. Fixes
                            gradient blow-up on a severely under-represented
                            class, but on real CICIDS-2017 data this was
                            observed to overcorrect the OTHER way: DoS/DDoS
                            (a substantial 14% of rows, not a tiny minority)
                            got enough of a boost that the model flipped
                            from "always predict Benign" to "mostly predict
                            DoS/DDoS", making net accuracy *worse* than the
                            trivial majority-class baseline.
      "balanced_sqrt"   -> sqrt-dampened inverse-frequency: weight ∝
                            sqrt(n / (k * count)) instead of the full ratio.
                            A gentler, commonly-used alternative that still
                            gives minority classes a boost without swinging
                            the decision boundary as hard. Try this first if
                            "balanced_capped" overcorrects.
    """
    if strategy == "none":
        return None

    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, 1.0, None)  # avoid div-by-zero for absent classes
    n_samples = counts.sum()
    weights = n_samples / (num_classes * counts)

    if strategy == "balanced_capped":
        min_w = weights.min()
        cap = min_w * max_ratio
        n_clipped = int((weights > cap).sum())
        if n_clipped:
            logger.info(
                "Clipping class weights for %d class(es) to %.2f "
                "(max_class_weight_ratio=%.1f) to prevent training collapse "
                "on severely under-represented classes.",
                n_clipped, cap, max_ratio,
            )
        weights = np.clip(weights, None, cap)
    elif strategy == "balanced_sqrt":
        weights = np.sqrt(weights)
    elif strategy != "balanced":
        raise ValueError(f"Unknown class_weight_strategy: {strategy}")

    weight_dict = {i: float(w) for i, w in enumerate(weights)}
    logger.info("Computed class weights (strategy=%s): %s", strategy, weight_dict)
    return weight_dict


# =============================================================================
# TensorFlow / device setup
# =============================================================================

def configure_device(device: str) -> None:
    """
    Configure TensorFlow's visible devices. Kept in one place so device
    selection never has to be duplicated (or forgotten) elsewhere.
    """
    import tensorflow as tf

    if device == "cpu":
        tf.config.set_visible_devices([], "GPU")
        logger.info("TensorFlow restricted to CPU (config.experiment.device=cpu).")
        return

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        logger.warning("device=gpu requested but no GPU visible to TensorFlow; falling back to CPU.")
        return

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logger.info("TensorFlow using %d GPU(s) with memory growth enabled.", len(gpus))


def reset_tf_session() -> None:
    """
    Clear the Keras/TF backend graph and free session state. Call this
    between agent (re)initializations within a single process — repeatedly
    building models without clearing sessions is a common source of
    creeping GPU/CPU memory growth over long training runs.
    """
    import tensorflow as tf
    tf.keras.backend.clear_session()


# =============================================================================
# Logging setup
# =============================================================================

def setup_logging(log_dir: str, name: str = "cm_mtd") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    if log.handlers:
        return log  # avoid duplicate handlers on repeated setup

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    log.addHandler(stream_handler)

    file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    file_handler.setFormatter(fmt)
    log.addHandler(file_handler)

    return log
