"""
Statistical analysis for CM-MTD experimental evaluation.
Implements paired t-test, Wilcoxon signed-rank test, Cohen's d effect size,
and 95% confidence intervals for multi-seed comparisons.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def paired_ttest(
    a: np.ndarray,
    b: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Perform paired Student's t-test between two methods.
    
    Tests H0: mean(a) == mean(b) against H1: mean(a) != mean(b).
    Appropriate when the same random seeds are used for both methods.
    
    Args:
        a: Performance scores for method A (n_seeds × n_episodes).
        b: Performance scores for method B (n_seeds × n_episodes).
        alpha: Significance level.
    
    Returns:
        Dictionary with t-statistic, p-value, and significance flag.
    """
    a_mean = np.mean(a, axis=-1) if a.ndim > 1 else a
    b_mean = np.mean(b, axis=-1) if b.ndim > 1 else b

    t_stat, p_value = stats.ttest_rel(a_mean, b_mean)

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
        "alpha": alpha,
        "test": "paired_t_test",
    }


def wilcoxon_test(
    a: np.ndarray,
    b: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).
    
    Args:
        a: Performance scores for method A.
        b: Performance scores for method B.
        alpha: Significance level.
    
    Returns:
        Dictionary with statistic, p-value, and significance flag.
    """
    a_mean = np.mean(a, axis=-1) if a.ndim > 1 else a
    b_mean = np.mean(b, axis=-1) if b.ndim > 1 else b

    try:
        stat, p_value = stats.wilcoxon(a_mean, b_mean)
    except ValueError:
        stat, p_value = 0.0, 1.0  # identical arrays

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
        "alpha": alpha,
        "test": "wilcoxon_signed_rank",
    }


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Cohen's d effect size.
    d = (mean_a - mean_b) / pooled_std
    
    Interpretation:
        |d| < 0.2  → negligible
        |d| < 0.5  → small
        |d| < 0.8  → medium
        |d| >= 0.8 → large
    
    Args:
        a: Scores for method A.
        b: Scores for method B.
    
    Returns:
        Cohen's d value.
    """
    a_mean = np.mean(a, axis=-1) if a.ndim > 1 else a
    b_mean = np.mean(b, axis=-1) if b.ndim > 1 else b

    pooled_std = np.sqrt((np.var(a_mean, ddof=1) + np.var(b_mean, ddof=1)) / 2)
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a_mean) - np.mean(b_mean)) / pooled_std)


def effect_size_label(d: float) -> str:
    """Classify Cohen's d into effect size categories."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def full_statistical_comparison(
    proposed: np.ndarray,
    baseline: np.ndarray,
    baseline_name: str = "baseline",
    alpha: float = 0.05,
) -> Dict[str, object]:
    """
    Perform full statistical comparison between proposed CM-MTD and a baseline.
    
    Args:
        proposed: DSR/scores for CM-MTD across seeds (shape: [n_seeds] or [n_seeds, n_episodes]).
        baseline: DSR/scores for baseline across seeds.
        baseline_name: Human-readable baseline name.
        alpha: Significance level.
    
    Returns:
        Comprehensive comparison dictionary.
    """
    prop_mean = np.mean(proposed, axis=-1) if proposed.ndim > 1 else proposed
    base_mean = np.mean(baseline, axis=-1) if baseline.ndim > 1 else baseline

    ci_prop = _compute_ci(prop_mean)
    ci_base = _compute_ci(base_mean)
    d = cohen_d(proposed, baseline)

    return {
        "proposed_cm_mtd": {
            "mean": float(np.mean(prop_mean)),
            "std": float(np.std(prop_mean, ddof=1)),
            "ci_lower": ci_prop[1],
            "ci_upper": ci_prop[2],
        },
        baseline_name: {
            "mean": float(np.mean(base_mean)),
            "std": float(np.std(base_mean, ddof=1)),
            "ci_lower": ci_base[1],
            "ci_upper": ci_base[2],
        },
        "paired_ttest": paired_ttest(proposed, baseline, alpha),
        "wilcoxon": wilcoxon_test(proposed, baseline, alpha),
        "cohen_d": d,
        "effect_size_label": effect_size_label(d),
        "improvement_pct": float(
            (np.mean(prop_mean) - np.mean(base_mean)) / max(abs(np.mean(base_mean)), 1e-8) * 100
        ),
    }


def _compute_ci(
    data: np.ndarray, confidence: float = 0.95
) -> Tuple[float, float, float]:
    """Internal: compute confidence interval."""
    n = len(data)
    mean = np.mean(data)
    if n < 2:
        return mean, mean, mean
    sem = stats.sem(data)
    h = sem * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return float(mean), float(mean - h), float(mean + h)


def summarize_all_baselines(
    results: Dict[str, np.ndarray],
    proposed_key: str = "CM_MTD",
) -> Dict[str, Dict]:
    """
    Summarize statistical comparisons for all baselines vs proposed.
    
    Args:
        results: Dict mapping method name → array of DSR scores [n_seeds, n_episodes].
        proposed_key: Key for the proposed CM-MTD method.
    
    Returns:
        Nested dict with all pairwise comparisons.
    """
    proposed = results[proposed_key]
    summary = {}

    for name, scores in results.items():
        if name == proposed_key:
            continue
        summary[name] = full_statistical_comparison(proposed, scores, name)

    return summary
