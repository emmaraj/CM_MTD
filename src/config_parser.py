"""
config_parser.py
-----------------
Configuration loading and dataset ingestion utilities for the CM-MTD
pipeline. Nothing here hardcodes a dataset's shape -- `input_dim` and
`num_classes` are always inferred from the loaded arrays (see
src/datasets/base.py::DatasetBundle), per the project's dataset-agnostic
requirement.

Dataset loading itself is delegated to src/datasets/ (one adapter per
dataset, routed through a small registry) -- see that package's module
docstring for why a single shared loader was the wrong call once a
second dataset (5G-NIDD) with its own directory contract and metadata
schema entered the picture. This module only knows "ask the registry for
whichever dataset config.experiment.dataset names."
"""

from __future__ import annotations

import os
import random
import logging
from typing import Any, Optional

import numpy as np
import yaml

from src.datasets import DatasetBundle, get_adapter

# Re-exported for any call sites (or REPL usage) still written against the
# pre-refactor `Dataset` name -- it's just the current DatasetBundle.
Dataset = DatasetBundle

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
    required_top_level = ["experiment", "datasets", "lstm", "transformer",
                           "network", "reward", "dqn", "ppo", "training"]
    missing = [k for k in required_top_level if k not in cfg]
    if missing:
        raise KeyError(f"config.yaml is missing required section(s): {missing}")

    exp = cfg["experiment"]
    for key in ("dataset", "predictor"):
        if key not in exp:
            raise KeyError(
                f"config.yaml's experiment: block is missing '{key}'. experiment.dataset and "
                f"experiment.predictor are the pipeline's two independent experimental "
                f"variables -- experiment.dataset selects a config.datasets.<name> block, "
                f"experiment.predictor selects config.lstm or config.transformer."
            )


def apply_cli_overrides(cfg: dict, dataset: Optional[str] = None, predictor: Optional[str] = None) -> dict:
    """
    Applies --dataset/--predictor CLI overrides on top of whatever
    config.yaml says, so comparison runs across datasets/architectures
    don't require hand-editing the file each time:

        cfg = apply_cli_overrides(load_config(args.config), args.dataset, args.predictor)

    Mutates and returns cfg for convenient chaining. Values are validated
    lazily -- active_dataset_cfg()/active_predictor_cfg() below raise a
    clear error if the override doesn't match anything configured.
    """
    if dataset is not None:
        cfg["experiment"]["dataset"] = dataset
    if predictor is not None:
        cfg["experiment"]["predictor"] = predictor
    return cfg


def active_dataset_cfg(cfg: dict) -> dict:
    """cfg['datasets'][cfg['experiment']['dataset']] -- this run's dataset sub-config."""
    dataset_name = cfg["experiment"]["dataset"]
    datasets_cfg = cfg.get("datasets", {})
    if dataset_name not in datasets_cfg:
        raise KeyError(
            f"experiment.dataset={dataset_name!r} but config.yaml's datasets: block has no "
            f"{dataset_name!r} entry. Configured datasets: {sorted(datasets_cfg.keys())}."
        )
    return datasets_cfg[dataset_name]


def active_predictor_cfg(cfg: dict) -> dict:
    """cfg[cfg['experiment']['predictor']] -- e.g. cfg['transformer'] or cfg['lstm']."""
    predictor = cfg["experiment"]["predictor"]
    if predictor not in cfg:
        raise KeyError(
            f"experiment.predictor={predictor!r} but config.yaml has no top-level "
            f"{predictor!r} section. Expected one of the sequence-model config blocks "
            f"(currently 'lstm' or 'transformer')."
        )
    return cfg[predictor]


def active_training_cfg(cfg: dict) -> dict:
    """
    cfg['training'] -- kept alongside active_dataset_cfg/active_predictor_cfg
    so call sites read uniformly ("the active X config") even though
    training: isn't currently split per-dataset/per-predictor the way
    lstm:/transformer: and datasets:<name> are.
    """
    return cfg["training"]


def get_run_paths(cfg: dict) -> dict:
    """
    Namespaces checkpoint/results output by dataset+predictor, e.g.
    checkpoints/cicids2017/transformer/, results/5g_nidd/lstm/. Without
    this, switching config.experiment.dataset or .predictor between runs
    would silently overwrite or reuse a previous run's checkpoints -- a
    real footgun now that dataset and predictor are two independent
    experimental variables (e.g. training on 5g_nidd right after
    cicids2017 could otherwise load stale CICIDS-2017 checkpoints into a
    "5G-NIDD" evaluation without any error).

    Also matches the results/<dataset_name>/<model_type>/ layout
    scripts/generate_figures.py already expects for its cross-dataset/
    cross-model auto-discovery figure pipeline.

    log_dir is left flat (un-namespaced) -- one process writing one log
    stream is fine either way, and it's convenient to `tail` a single
    file across an experimentation session that touches multiple
    dataset/predictor combinations.
    """
    exp_cfg = cfg["experiment"]
    dataset_name = exp_cfg["dataset"]
    predictor = exp_cfg["predictor"]
    run_subdir = os.path.join(dataset_name, predictor)
    return {
        "log_dir": exp_cfg["log_dir"],
        "checkpoint_dir": os.path.join(exp_cfg["checkpoint_dir"], run_subdir),
        "results_dir": os.path.join(exp_cfg["results_dir"], run_subdir),
        "run_subdir": run_subdir,
    }


def set_global_seed(seed: int) -> None:
    """Seed python, numpy, and (if importable) torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# =============================================================================
# Dataset loading -- delegates to src/datasets/'s adapter registry
# =============================================================================

def load_dataset(cfg: dict) -> DatasetBundle:
    """
    Loads whichever dataset config.experiment.dataset names, via that
    dataset's adapter (src/datasets/). Returns a DatasetBundle with
    input_dim/num_classes inferred from the arrays, never hardcoded --
    see src/datasets/base.py.
    """
    dataset_name = cfg["experiment"]["dataset"]
    adapter = get_adapter(dataset_name)
    dataset_cfg = active_dataset_cfg(cfg)
    logger.info("Loading dataset %r via %s", dataset_name, type(adapter).__name__)
    bundle = adapter.load(dataset_cfg)
    logger.info(
        "Dataset %r ready: input_dim=%d, num_classes=%d, classes=%s, "
        "train=%d val=%s test=%d, reconnaissance_class=%r(id=%d), flooding_class=%r(id=%d)",
        dataset_name, bundle.input_dim, bundle.num_classes, bundle.class_names,
        len(bundle.X_train), len(bundle.X_val) if bundle.X_val is not None else "n/a", len(bundle.X_test),
        bundle.reconnaissance_class_name, bundle.reconnaissance_class,
        bundle.flooding_class_name, bundle.flooding_class,
    )
    return bundle


def _stratified_head(X: np.ndarray, y: np.ndarray, max_samples: int) -> tuple:
    """
    Deterministically take the first-occurring rows per class, interleaved
    in original order, until max_samples rows are collected. No randomness
    is used -- this keeps dataset truncation reproducible and consistent
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


def build_env_row_cache(dataset: DatasetBundle, dataset_cfg: dict) -> tuple:
    """
    Builds the (possibly truncated) X/y view the RL environment cycles
    through for its per-node row assignment. This is the ONLY place
    max_samples/truncation_strategy apply -- Stage-2 sequence-model
    training always sees the full dataset via `dataset.X_train`/
    `dataset.y_train` directly. Row order/continuity doesn't matter here
    the way it does for LSTM/Transformer sequence windows, since
    environment.py already assigns rows to nodes via an artificial
    deterministic round-robin, not genuine per-node temporal adjacency.

    dataset_cfg is this dataset's own config.datasets.<name> block (see
    config_parser.active_dataset_cfg) -- max_samples/truncation_strategy
    now live per-dataset (each dataset has its own size/imbalance
    profile) rather than in one shared top-level `data:` block.
    """
    X_train, y_train = dataset.X_train, dataset.y_train
    max_samples = dataset_cfg.get("max_samples")
    if max_samples is not None and len(X_train) > max_samples:
        strategy = dataset_cfg.get("truncation_strategy", "head")
        if strategy == "stratified_head":
            X_train, y_train = _stratified_head(X_train, y_train, max_samples)
        elif strategy == "head":
            X_train, y_train = X_train[:max_samples], y_train[:max_samples]
        else:
            raise ValueError(f"Unknown truncation_strategy: {strategy}")
        logger.info(
            "Environment row cache truncated to %d rows via '%s' strategy "
            "(bounds memory for the RL loop; Stage-2 training is unaffected "
            "and uses the full dataset).",
            len(X_train), strategy,
        )
    return X_train, y_train


# =============================================================================
# Class weighting (guards against the LSTM/Transformer-collapse failure
# mode seen with CICIDS-2017's severely under-represented minority class)
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
                            observed to overcorrect the OTHER way: DDoS
                            (a substantial ~8-13% of rows, not a tiny
                            minority) got enough of a boost that the model
                            flipped from "always predict Benign" to "mostly
                            predict DDoS", making net accuracy *worse* than
                            the trivial majority-class baseline.
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
# PyTorch / device setup
# =============================================================================

def configure_device(device: str):
    """
    Resolves config.experiment.device ("cpu" | "gpu") to a torch.device,
    which callers then pass explicitly into every model constructor
    (LSTMAttackPredictor/TransformerAttackPredictor/DQNAgent/PPOAgent all
    take a `device=` kwarg). This is a real API change from the Keras
    version, which configured TF's GPU visibility as global process
    state and never threaded a device object through call sites --
    PyTorch has no equivalent global "restrict visible devices" switch,
    so being explicit here is the idiomatic PyTorch way rather than a
    workaround.

    Returns torch.device("cpu") unconditionally for device="cpu". For
    device="gpu", returns a CUDA device if one is visible, else falls
    back to CPU with a warning (matching the Keras version's fallback
    behavior for device="gpu" with no GPU present).
    """
    import torch

    if device == "cpu":
        logger.info("Using CPU (config.experiment.device=cpu).")
        return torch.device("cpu")

    if torch.cuda.is_available():
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
        return torch.device("cuda")

    logger.warning("device=gpu requested but no CUDA device visible to PyTorch; falling back to CPU.")
    return torch.device("cpu")


def reset_torch_session() -> None:
    """
    Frees cached CUDA memory and runs a Python GC pass between agent
    (re)initializations within a single process. PyTorch has no
    equivalent to Keras's persistent global default graph/session, so
    this is less critical here than `reset_tf_session` was for the Keras
    version -- but repeatedly building and discarding nn.Module instances
    can still leave cached CUDA allocator blocks around, so this is kept
    as a cheap, explicit reset point at the same call site the Keras
    version used (train_attack_predictor, before building each new
    predictor).
    """
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
