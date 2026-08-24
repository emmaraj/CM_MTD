"""
metrics.py
----------
Fairness, Trust, and Energy evaluation for the attack predictor (LSTM or
Transformer, Stage 2) and the RL defense policy. These are not metrics
the paper reports -- they're added to assess trade-offs of the
architecture change (LSTM -> Transformer) beyond raw predictive accuracy,
per the project's methodology report.

None of these are exotic: each is a standard, citable technique adapted
to this project's specific outputs (a 3-class event predictor feeding a
per-node MTD policy), not a novel metric invented for this report.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("cm_mtd")


# =============================================================================
# Fairness
#
# There's no demographic attribute in network traffic, so "fairness" here
# is adapted to the two groupings that actually matter for this system:
#   (a) traffic CLASSES -- does the predictor perform equitably across
#       Benign/Scan-Infiltration/DDoS, or does it sacrifice minority-class
#       performance for majority-class accuracy? (this was the central
#       failure mode debugged throughout this project, so it's a directly
#       meaningful axis here, not a stretch)
#   (b) network NODES -- does the resulting MTD policy defend all nodes
#       roughly equally, or does it concentrate protection on some nodes
#       while leaving others comparatively exposed?
# =============================================================================

def class_fairness(per_class_recall: dict) -> dict:
    """
    Fairness across traffic classes, from a dict {class_id: recall}
    (e.g. LSTMAttackPredictor.compute_fidelity()['per_class_accuracy']).

    Returns:
      min_recall, max_recall: the worst- and best-served classes.
      recall_gap: max - min (0 = perfectly equitable; closer to 1 = one
        class is being sacrificed for another -- this is exactly the
        gap that was ~1.0 during this project's majority-class-collapse
        failures, e.g. Benign=1.00 vs DDoS=0.00).
      equity_ratio: min / max (1 = perfectly equitable; 0 = total
        neglect of the worst-served class). Undefined (returned as None)
        if max_recall is 0.
    """
    recalls = [v for v in per_class_recall.values() if v is not None]
    if not recalls:
        return {"min_recall": None, "max_recall": None, "recall_gap": None, "equity_ratio": None}
    min_r, max_r = min(recalls), max(recalls)
    return {
        "min_recall": min_r,
        "max_recall": max_r,
        "recall_gap": max_r - min_r,
        "equity_ratio": (min_r / max_r) if max_r > 0 else None,
    }


def node_fairness(per_node_dsr: np.ndarray) -> dict:
    """
    Fairness across network nodes, from a per-node Defense Success Ratio
    array (Eq. 19 computed individually per node instead of network-wide).

    Returns mean/std/coefficient-of-variation and a Jain's fairness index
    (a standard fairness measure from resource-allocation/networking
    literature, J = (sum x)^2 / (n * sum x^2); J=1 means every node gets
    identical DSR, J=1/n means all protection is concentrated on one node).
    """
    x = np.asarray(per_node_dsr, dtype=np.float64)
    if len(x) == 0:
        return {"mean": None, "std": None, "cv": None, "jains_index": None}
    mean = float(np.mean(x))
    std = float(np.std(x))
    cv = float(std / mean) if mean > 0 else None
    jains = float((np.sum(x) ** 2) / (len(x) * np.sum(x ** 2))) if np.sum(x ** 2) > 0 else None
    return {"mean": mean, "std": std, "cv": cv, "jains_index": jains}


# =============================================================================
# Trust
#
# "Trust" is operationalized via three standard, independent signals:
#   (a) calibration -- does predicted confidence match actual accuracy?
#       (Guo et al., 2017, "On Calibration of Modern Neural Networks")
#   (b) attention entropy -- for the Transformer specifically, how
#       concentrated vs. diffuse is its attention over recent history
#       (a directly inspectable signal an LSTM's hidden state doesn't
#       offer without a separate post-hoc explainability method)
#   (c) prediction stability -- does a semantically-irrelevant perturbation
#       (reordering two adjacent Benign events) change the prediction?
#       A trustworthy model shouldn't be brittle to noise that carries no
#       real signal.
# =============================================================================

def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> dict:
    """
    Expected Calibration Error (ECE): bins predictions by confidence
    (max predicted probability) and compares each bin's average confidence
    to its actual accuracy. ECE=0 is perfect calibration; higher is worse.
    Also returns per-bin detail for a reliability diagram.
    """
    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    correct = (predictions == y_true).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = []
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        count = int(in_bin.sum())
        if count == 0:
            bins.append({"range": (float(lo), float(hi)), "count": 0, "avg_confidence": None, "accuracy": None})
            continue
        avg_conf = float(confidences[in_bin].mean())
        acc = float(correct[in_bin].mean())
        ece += (count / n) * abs(avg_conf - acc)
        bins.append({"range": (float(lo), float(hi)), "count": count, "avg_confidence": avg_conf, "accuracy": acc})

    return {"ece": float(ece), "bins": bins}


def attention_entropy(attention_weights: np.ndarray) -> dict:
    """
    Mean normalized entropy of attention distributions, from
    TransformerAttackPredictor.get_attention_weights() output
    (batch, heads, seq_len, seq_len).

    Entropy is computed per query position's attention distribution over
    key positions, then normalized by log(seq_len) so the result is in
    [0, 1] regardless of sequence length: 0 = fully concentrated on one
    past position (maximally interpretable: "this one event drove the
    prediction"), 1 = uniform over all positions (attention carries no
    information about which past event mattered).
    """
    eps = 1e-12
    p = attention_weights + eps
    ent = -np.sum(p * np.log(p), axis=-1)  # (batch, heads, seq_len)
    max_ent = np.log(attention_weights.shape[-1])
    normalized = ent / max_ent
    return {
        "mean_normalized_entropy": float(np.mean(normalized)),
        "std_normalized_entropy": float(np.std(normalized)),
        "min_normalized_entropy": float(np.min(normalized)),
        "max_normalized_entropy": float(np.max(normalized)),
    }


def prediction_stability(predictor, event_labels: np.ndarray, target_labels: np.ndarray,
                          benign_class: int = 0, n_samples: int = 500, seed: int = 42) -> dict:
    """
    Robustness/consistency check: for windows containing at least two
    adjacent Benign (non-informative) events, swap that adjacent pair and
    re-predict. A trustworthy model's prediction shouldn't flip from a
    perturbation that carries no real signal -- reordering two routine
    Benign events doesn't change "what's been happening recently" in any
    meaningful sense. Returns the fraction of predictions that flipped.
    """
    windows, targets = predictor.build_sliding_windows(event_labels, target_labels)
    rng = np.random.RandomState(seed)

    candidates = []
    for i in range(len(windows)):
        w = windows[i]
        adj_benign = [j for j in range(len(w) - 1) if w[j] == benign_class and w[j + 1] == benign_class]
        if adj_benign:
            candidates.append((i, rng.choice(adj_benign)))
    if not candidates:
        return {"n_tested": 0, "flip_rate": None}

    idx = rng.choice(len(candidates), size=min(n_samples, len(candidates)), replace=False)
    sample = [candidates[i] for i in idx]

    orig_windows = np.stack([windows[i] for i, _ in sample])
    perturbed_windows = orig_windows.copy()
    for k, (_, j) in enumerate(sample):
        perturbed_windows[k, j], perturbed_windows[k, j + 1] = perturbed_windows[k, j + 1], perturbed_windows[k, j]

    orig_preds = predictor.predict_next_events(orig_windows)
    perturbed_preds = predictor.predict_next_events(perturbed_windows)
    flips = int(np.sum(orig_preds != perturbed_preds))

    return {"n_tested": len(sample), "n_flipped": flips, "flip_rate": flips / len(sample)}


# =============================================================================
# Energy
#
# Wraps codecarbon (github.com/mlco2/codecarbon, a standard Python
# library for estimating energy consumption / CO2 emissions from
# CPU/GPU/RAM usage) when available, with a manual wall-clock-time-based
# fallback so this doesn't add a hard dependency or fail on systems where
# codecarbon can't read hardware power counters (e.g. some containers/VMs
# block RAPL access).
# =============================================================================

_ASSUMED_AVG_POWER_WATTS = 65.0  # rough average CPU TDP, used ONLY by the fallback estimator


@dataclass
class EnergyReport:
    wall_clock_seconds: float
    estimated_energy_kwh: Optional[float]
    estimated_co2_kg: Optional[float]
    method: str  # "codecarbon" | "fallback_estimate"
    trainable_params: Optional[int] = None
    inference_ms_per_call: Optional[float] = None


@contextmanager
def track_energy(label: str = "run"):
    """
    Usage:
        with track_energy("transformer_training") as report:
            predictor.fit(...)
        print(report.wall_clock_seconds, report.estimated_energy_kwh)

    report is populated on context exit. Falls back to a rough wall-clock
    x assumed-average-CPU-power estimate if codecarbon isn't usable (e.g.
    no permission to read RAPL energy counters) -- fallback numbers are
    explicitly a rough estimate, not a measurement, and are labeled as such
    in the returned report so they're never mistaken for real readings.
    """
    report = EnergyReport(wall_clock_seconds=0.0, estimated_energy_kwh=None,
                           estimated_co2_kg=None, method="fallback_estimate")
    tracker = None
    start = time.time()
    try:
        from codecarbon import EmissionsTracker
        tracker = EmissionsTracker(project_name=f"cm_mtd_{label}", log_level="error",
                                    save_to_file=False, allow_multiple_runs=True)
        tracker.start()
    except Exception as e:
        logger.warning("codecarbon unavailable/unusable (%s); falling back to a rough "
                        "wall-clock-time x assumed-average-CPU-power estimate, NOT a "
                        "real hardware energy measurement.", e)
        tracker = None

    try:
        yield report
    finally:
        elapsed = time.time() - start
        report.wall_clock_seconds = elapsed
        if tracker is not None:
            try:
                emissions_kg = tracker.stop()
                report.estimated_co2_kg = float(emissions_kg) if emissions_kg is not None else None
                # codecarbon tracks energy internally; expose it via its own
                # public attribute rather than re-deriving it.
                energy_kwh = getattr(tracker, "_total_energy", None)
                report.estimated_energy_kwh = float(energy_kwh.kWh) if energy_kwh is not None else None
                report.method = "codecarbon"
            except Exception as e:
                logger.warning("codecarbon failed to report results (%s); using fallback estimate.", e)
                report.estimated_energy_kwh = (_ASSUMED_AVG_POWER_WATTS * elapsed / 3600.0) / 1000.0
        else:
            report.estimated_energy_kwh = (_ASSUMED_AVG_POWER_WATTS * elapsed / 3600.0) / 1000.0


def count_trainable_params(keras_model) -> int:
    return int(sum(np.prod(v.shape) for v in keras_model.trainable_variables))


def benchmark_inference_latency(predict_fn, sample_input, n_calls: int = 200, warmup: int = 10) -> float:
    """
    Mean milliseconds per call for a single-window prediction. Reuses the
    same warmup-then-time methodology used earlier in this project to
    validate the tf.function performance fixes -- first call(s) pay
    one-time graph-tracing cost, which must be excluded from the average.
    """
    for _ in range(warmup):
        predict_fn(sample_input)
    start = time.time()
    for _ in range(n_calls):
        predict_fn(sample_input)
    elapsed = time.time() - start
    return (elapsed / n_calls) * 1000.0
