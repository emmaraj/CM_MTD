"""
visualize_results.py
====================
Standalone visualization script for CM-MTD experiments.

Reads ONLY from the CSV / JSON log files produced by train_and_eval.py.
Does NOT import any CM-MTD module or generate mock data.

Figures produced (matching the paper):
  Fig 7  → lstm_fidelity_loss_{mode}.png    (per-mode fidelity + loss curves)
  Fig 8  → confusion_matrix_{mode}.png      (per-class confusion heat-maps)
  Fig 9a → dsr_comparison_{mode}.png        (DSR line plots per attack type)
  Fig 9d → dsr_bar_all_modes.png            (bar chart across all modes)
  Fig 10 → network_performance_{mode}.png   (RTT and PLR over time)
  Fig 11 → convergence_{mode}.png           (upper + lower reward curves)

Usage
-----
    python visualize_results.py --log_dir ./logs --output_dir ./figures

All figures are saved as high-resolution PNG files; no interactive windows
are opened (Agg backend is forced).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import uniform_filter1d

warnings.filterwarnings("ignore")

# ── visual style ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = {
    "CM-MTD"    : "#E63946",
    "RRT+FRVM"  : "#2A9D8F",
    "DQ-RM+FRVM": "#457B9D",
    "no_mutation": "#6C757D",
    "train"     : "#E76F51",
    "test"      : "#264653",
}
MODES = ["direct_ddos", "crossfire_ddos", "cicids2017"]
MODE_LABELS = {
    "direct_ddos"   : "Direct DDoS + Sequential Scanning",
    "crossfire_ddos": "Crossfire DDoS + Sequential Scanning",
    "cicids2017"    : "CICIDS-2017",
}
EVENT_LABELS = ["BENIGN", "INFILTRATION", "DoS/DDoS"]
SMOOTH_WIN = 51   # smoothing window for line plots


# ===========================================================================
# Utility helpers
# ===========================================================================

def _smooth(series: np.ndarray, window: int = SMOOTH_WIN) -> np.ndarray:
    """Moving-average smoother — same visual effect as error-shadow plots."""
    w = min(window, len(series))
    return uniform_filter1d(series.astype(float), size=w, mode="nearest")


def _save(fig: plt.Figure, path: str):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        print(f"  [WARN] File not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_json(path: str) -> dict:
    if not os.path.isfile(path):
        print(f"  [WARN] File not found: {path}")
        return {}
    with open(path) as f:
        return json.load(f)


def _require_columns(df: pd.DataFrame, cols: list, src: str) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"  [WARN] {src} missing columns: {missing}")
        return False
    return True


# ===========================================================================
# Figure 7 — LSTM fidelity & loss
# ===========================================================================

def plot_lstm_fidelity_loss(log_dir: str, out_dir: str):
    """
    Reproduces Fig. 7 from the paper:
    Left panel  — prediction accuracy fidelity vs episode.
    Right panel — training loss vs episode.
    One figure per dataset mode.
    """
    print("\n[Fig 7] LSTM fidelity / loss …")

    for mode in MODES:
        csv_path = os.path.join(log_dir, f"lstm_training_log_{mode}.csv")
        df = _load_csv(csv_path)
        if df.empty:
            continue
        if not _require_columns(df, ["episode", "phase", "loss", "fidelity"], csv_path):
            continue

        df["loss"]     = pd.to_numeric(df["loss"],     errors="coerce")
        df["fidelity"] = pd.to_numeric(df["fidelity"], errors="coerce")

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        fig.suptitle(f"Attack Prediction Performance — {MODE_LABELS[mode]}", fontsize=12)

        for phase in ("train", "test"):
            sub = df[df["phase"] == phase].sort_values("episode")
            if sub.empty:
                continue
            lbl = "Train" if phase == "train" else "Test"
            col = PALETTE[phase]
            axes[0].plot(
                sub["episode"], _smooth(sub["fidelity"].values),
                label=f"{lbl} Fidelity", color=col, linewidth=2
            )
            axes[1].plot(
                sub["episode"], _smooth(sub["loss"].values),
                label=f"{lbl} Loss", color=col, linewidth=2, linestyle="--"
            )

        for ax, ylabel in zip(axes, ["Prediction Accuracy Fidelity", "Loss"]):
            ax.set_xlabel("Episode")
            ax.set_ylabel(ylabel)
            ax.legend(framealpha=0.8)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        axes[0].set_ylim(0, 1.05)
        plt.tight_layout()
        _save(fig, os.path.join(out_dir, f"lstm_fidelity_loss_{mode}.png"))


# ===========================================================================
# Figure 8 — Confusion matrices
# ===========================================================================

def plot_confusion_matrices(log_dir: str, out_dir: str):
    """
    Reproduces Fig. 8: heat-map confusion matrices on the test set
    for each dataset mode.
    """
    print("\n[Fig 8] Confusion matrices …")

    cm_data = _load_json(os.path.join(log_dir, "confusion_matrices.json"))
    if not cm_data:
        return

    for mode, algo_dict in cm_data.items():
        for algo_name, metrics in algo_dict.items():
            cm = np.array(metrics.get("matrix", []))
            if cm.size == 0:
                continue

            n_cls = cm.shape[0]
            labels = EVENT_LABELS[:n_cls]

            # Normalise rows to get recall-based heat-map
            row_sums = cm.sum(axis=1, keepdims=True).clip(1, None)
            cm_norm  = cm / row_sums

            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            fig.suptitle(
                f"Confusion Matrix ({algo_name}) — {MODE_LABELS.get(mode, mode)}",
                fontsize=12,
            )

            # Raw counts
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                ax=axes[0], cbar=True,
            )
            axes[0].set_title("Counts")
            axes[0].set_xlabel("Predicted")
            axes[0].set_ylabel("Actual")

            # Normalised
            sns.heatmap(
                cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                ax=axes[1], cbar=True, vmin=0, vmax=1,
            )
            axes[1].set_title("Row-normalised (Recall per class)")
            axes[1].set_xlabel("Predicted")
            axes[1].set_ylabel("Actual")

            # Print per-class metrics
            prec = metrics.get("precision", [])
            rec  = metrics.get("recall",    [])
            f1   = metrics.get("f1",        [])
            for c_idx, c_name in enumerate(labels):
                if c_idx < len(prec):
                    print(
                        f"    [{mode}][{c_name}]  "
                        f"P={prec[c_idx]:.3f}  R={rec[c_idx]:.3f}  F1={f1[c_idx]:.3f}"
                    )

            plt.tight_layout()
            _save(fig, os.path.join(out_dir, f"confusion_matrix_{mode}.png"))


# ===========================================================================
# Figure 9 — Defense performance
# ===========================================================================

def plot_defense_performance(log_dir: str, out_dir: str):
    """
    Reproduces Fig. 9:
    (a–c)  DSR vs episode — one subplot per mode.
    (d)    Bar chart — final converged DSR across all modes.
    """
    print("\n[Fig 9] Defense performance …")

    df = _load_csv(os.path.join(log_dir, "defense_log.csv"))
    if df.empty:
        return
    if not _require_columns(df, ["episode", "algorithm", "attack_type", "dsr"], "defense_log.csv"):
        return

    df["dsr"]     = pd.to_numeric(df["dsr"],     errors="coerce")
    df["episode"] = pd.to_numeric(df["episode"], errors="coerce")

    algos = ["CM-MTD", "RRT+FRVM", "DQ-RM+FRVM"]

    # --- Per-mode line plots (Figs 9a–9c) ---
    for mode in MODES:
        sub = df[df["attack_type"] == mode]
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.set_title(f"Defense Success Ratio — {MODE_LABELS.get(mode, mode)}", fontsize=12)

        for algo in algos:
            asub = sub[sub["algorithm"] == algo].sort_values("episode")
            if asub.empty:
                continue
            smoothed = _smooth(asub["dsr"].values)
            std_val  = asub["dsr"].rolling(SMOOTH_WIN, min_periods=1).std().fillna(0).values
            ax.plot(asub["episode"], smoothed,
                    label=algo, color=PALETTE[algo], linewidth=2)
            ax.fill_between(
                asub["episode"],
                (smoothed - std_val).clip(0, 100),
                (smoothed + std_val).clip(0, 100),
                color=PALETTE[algo], alpha=0.15,
            )

        ax.set_xlabel("Episode")
        ax.set_ylabel("Defense Success Ratio (%)")
        ax.set_ylim(75, 102)
        ax.legend(framealpha=0.85)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        plt.tight_layout()
        _save(fig, os.path.join(out_dir, f"dsr_comparison_{mode}.png"))

    # --- Bar chart (Fig 9d) — converged DSR per mode ---
    converge_frac = 0.25    # use last 25 % of episodes as "converged"
    bar_data: dict = {algo: [] for algo in algos}
    bar_errors: dict = {algo: [] for algo in algos}

    for mode in MODES:
        sub = df[df["attack_type"] == mode]
        max_ep = sub["episode"].max() if not sub.empty else 1
        thresh = max_ep * (1 - converge_frac)
        for algo in algos:
            asub = sub[(sub["algorithm"] == algo) & (sub["episode"] >= thresh)]
            bar_data[algo].append(asub["dsr"].mean() if not asub.empty else 0.0)
            bar_errors[algo].append(asub["dsr"].std()  if not asub.empty else 0.0)

    x      = np.arange(len(MODES))
    width  = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("Converged Defense Success Ratio (All Attack Types)", fontsize=12)

    for i, algo in enumerate(algos):
        ax.bar(
            x + (i - 1) * width,
            bar_data[algo],
            width,
            yerr=bar_errors[algo],
            label=algo,
            color=PALETTE[algo],
            capsize=4,
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS[m] for m in MODES], fontsize=9)
    ax.set_ylabel("Defense Success Ratio (%)")
    ax.set_ylim(70, 105)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(framealpha=0.85)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "dsr_bar_all_modes.png"))


# ===========================================================================
# Figure 10 — Network performance (RTT & PLR)
# ===========================================================================

def plot_network_performance(log_dir: str, out_dir: str):
    """
    Reproduces Fig. 10:
    RTT (ms) and PLR (%) when CM-MTD is active vs no mutation.
    """
    print("\n[Fig 10] Network performance …")

    df = _load_csv(os.path.join(log_dir, "network_perf_log.csv"))
    if df.empty:
        return
    if not _require_columns(df, ["episode", "algorithm", "attack_type", "rtt_ms", "plr_pct"],
                            "network_perf_log.csv"):
        return

    df["rtt_ms"]  = pd.to_numeric(df["rtt_ms"],  errors="coerce")
    df["plr_pct"] = pd.to_numeric(df["plr_pct"], errors="coerce")
    df["episode"] = pd.to_numeric(df["episode"], errors="coerce")

    for mode in MODES:
        sub = df[df["attack_type"] == mode]
        if sub.empty:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        fig.suptitle(f"Network Performance — {MODE_LABELS.get(mode, mode)}", fontsize=12)

        for algo, col, label in [
            ("CM-MTD",     PALETTE["CM-MTD"],     "CM-MTD"),
            ("no_mutation", PALETTE["no_mutation"], "No Mutation"),
        ]:
            asub = sub[sub["algorithm"] == algo].sort_values("episode")
            if asub.empty:
                continue
            rtt_s = _smooth(asub["rtt_ms"].values,  window=30)
            plr_s = _smooth(asub["plr_pct"].values, window=30)
            axes[0].plot(asub["episode"], rtt_s,  label=label, color=col, linewidth=2)
            axes[1].plot(asub["episode"], plr_s,  label=label, color=col, linewidth=2)

        for ax, ylabel in zip(axes, ["RTT (ms)", "Packet Loss Ratio (%)"]):
            ax.set_xlabel("Episode")
            ax.set_ylabel(ylabel)
            ax.legend(framealpha=0.85)
            ax.set_ylim(bottom=0)

        plt.tight_layout()
        _save(fig, os.path.join(out_dir, f"network_performance_{mode}.png"))


# ===========================================================================
# Figure 11 — Convergence (upper + lower rewards)
# ===========================================================================

def plot_convergence(log_dir: str, out_dir: str):
    """
    Reproduces Fig. 11:
    Upper-layer (DQN) and lower-layer (PPO) reward curves per mode.
    """
    print("\n[Fig 11] Convergence …")

    df = _load_csv(os.path.join(log_dir, "convergence_log.csv"))
    if df.empty:
        return
    if not _require_columns(df, ["episode", "layer", "attack_type", "reward"],
                            "convergence_log.csv"):
        return

    df["reward"]  = pd.to_numeric(df["reward"],  errors="coerce")
    df["episode"] = pd.to_numeric(df["episode"], errors="coerce")

    for mode in MODES:
        sub = df[df["attack_type"] == mode]
        if sub.empty:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        fig.suptitle(f"HDRL Convergence — {MODE_LABELS.get(mode, mode)}", fontsize=12)

        for ax, layer, title in [
            (axes[0], "upper", "Upper Layer (DQN) — Episode Reward"),
            (axes[1], "lower", "Lower Layer (PPO) — Step Reward"),
        ]:
            lsub = sub[sub["layer"] == layer].sort_values("episode")
            if lsub.empty:
                ax.set_visible(False)
                continue
            smoothed = _smooth(lsub["reward"].values)
            std_arr  = (
                lsub["reward"]
                .rolling(SMOOTH_WIN, min_periods=1)
                .std()
                .fillna(0)
                .values
            )
            ax.plot(lsub["episode"], smoothed, color=PALETTE["CM-MTD"], linewidth=2)
            ax.fill_between(
                lsub["episode"],
                smoothed - std_arr,
                smoothed + std_arr,
                color=PALETTE["CM-MTD"], alpha=0.20,
            )
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Reward")

        plt.tight_layout()
        _save(fig, os.path.join(out_dir, f"convergence_{mode}.png"))


# ===========================================================================
# Summary statistics table
# ===========================================================================

def print_summary(log_dir: str):
    print("\n[Summary] Final statistics:")
    summary = _load_json(os.path.join(log_dir, "final_summary.json"))
    if not summary:
        return
    header = f"{'Mode':<22} {'CM-MTD':>10} {'RRT+FRVM':>12} {'DQ-RM+FRVM':>14}"
    print(f"  {header}")
    print(f"  {'-'*60}")
    for mode, vals in summary.items():
        cm   = vals.get("CM-MTD_final_dsr",     0.0)
        rrt  = vals.get("RRT+FRVM_final_dsr",   0.0)
        dqrm = vals.get("DQ-RM+FRVM_final_dsr", 0.0)
        row = f"  {MODE_LABELS.get(mode, mode):<22}  {cm:>7.1f}%  {rrt:>10.1f}%  {dqrm:>12.1f}%"
        print(row)


# ===========================================================================
# Entry point
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="CM-MTD visualizer — generates all paper figures from log files."
    )
    p.add_argument("--log_dir",    type=str, default="./logs",
                   help="Directory containing CSV / JSON log files.")
    p.add_argument("--output_dir", type=str, default="./figures",
                   help="Directory where PNG figures will be saved.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reading logs from : {args.log_dir}")
    print(f"Saving figures to : {args.output_dir}")

    plot_lstm_fidelity_loss   (args.log_dir, args.output_dir)
    plot_confusion_matrices   (args.log_dir, args.output_dir)
    plot_defense_performance  (args.log_dir, args.output_dir)
    plot_network_performance  (args.log_dir, args.output_dir)
    plot_convergence          (args.log_dir, args.output_dir)
    print_summary             (args.log_dir)

    print(f"\nAll figures written to {args.output_dir}/")


if __name__ == "__main__":
    main()
