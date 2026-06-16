"""
Metrics computation for CM-MTD evaluation.
Implements prediction metrics (Eq. 18) and defense metrics (Eq. 19) from the paper.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    average: str = "macro",
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    Implements prediction accuracy fidelity (Eq. 18 from paper).
    
    Fidelity = Σ_{i∈N} 𝕐(p_i, y_i) / |N|
    where 𝕐(p_i, y_i) = 1 if p_i == y_i else 0.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Optional list of class names for per-class metrics.
        average: Averaging strategy ('macro', 'weighted', 'micro').
    
    Returns:
        Dictionary with accuracy, precision, recall, F1, and per-class metrics.
    """
    metrics = {
        "fidelity": float(accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    # Per-class metrics
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    metrics["per_class"] = {
        "precision": per_class_precision.tolist(),
        "recall": per_class_recall.tolist(),
        "f1": per_class_f1.tolist(),
    }

    if class_names:
        metrics["class_names"] = class_names

    return metrics


def compute_dsr(
    n_nodes_scanned: np.ndarray,
    n_switches_compromised: np.ndarray,
    total_nodes_scanned: np.ndarray,
    total_switches_on_route: np.ndarray,
) -> np.ndarray:
    """
    Compute Defense Success Ratio (DSR) — Eq. 19 from paper.
    
    DSR = (1 - (Σ N^s_k + Σ N^d_k) / (Σ L^s_k + Σ L^d_k)) × 100%
    
    Args:
        n_nodes_scanned:       N^s_k — nodes compromised by scanning per episode.
        n_switches_compromised: N^d_k — switches compromised by DDoS per episode.
        total_nodes_scanned:   L^s_k — total scanned network nodes.
        total_switches_on_route: L^d_k — total switches on transmission routes.
    
    Returns:
        DSR values (%) for each episode.
    """
    numerator = n_nodes_scanned + n_switches_compromised
    denominator = total_nodes_scanned + total_switches_on_route
    # Avoid division by zero
    safe_denom = np.where(denominator > 0, denominator, 1)
    dsr = (1 - numerator / safe_denom) * 100.0
    return np.clip(dsr, 0.0, 100.0)


def moving_average(values: np.ndarray, window: int = 100) -> np.ndarray:
    """Compute moving average for smoothing training curves."""
    if len(values) < window:
        return values
    weights = np.ones(window) / window
    return np.convolve(values, weights, mode="valid")


def compute_confidence_interval(
    data: np.ndarray, confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute mean ± confidence interval using Student's t-distribution.
    
    Args:
        data: Array of values across multiple seeds.
        confidence: Confidence level (e.g., 0.95 for 95% CI).
    
    Returns:
        Tuple of (mean, lower_bound, upper_bound).
    """
    from scipy import stats
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    h = sem * stats.t.ppf((1 + confidence) / 2., n - 1)
    return float(mean), float(mean - h), float(mean + h)


def compute_std_error(data: np.ndarray) -> float:
    """Compute standard error of the mean."""
    return float(np.std(data, ddof=1) / np.sqrt(len(data)))
