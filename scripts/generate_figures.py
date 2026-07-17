"""
generate_figures.py
--------------------
Reads the metrics saved by src/main.py (LSTM training history, confusion
data, RL reward/DSR curves) and renders paper-comparable figures:

  fig7_lstm_fidelity_loss.png   <- Fig. 7: prediction fidelity + loss vs. epoch
  fig8_confusion_matrix.png     <- Fig. 8: confusion matrix on the test set
  fig9_dsr_over_training.png    <- Fig. 9-style: DSR (Eq. 19) over training
  fig11_convergence.png         <- Fig. 11: upper/lower layer reward convergence

Fig. 10 (RTT / packet-loss network performance) is intentionally NOT
generated here -- see the printed note below for why.

Usage:
    python3 -m src.main --config config/config.yaml --mode all   # produces the data
    python3 scripts/generate_figures.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def fig7_lstm_fidelity_loss(results_dir: str, out_dir: str) -> bool:
    path = os.path.join(results_dir, "lstm_history.json")
    if not os.path.exists(path):
        print(f"[skip] {path} not found -- run --mode train_lstm first.")
        return False

    with open(path) as f:
        history = json.load(f)

    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    acc_key = "accuracy" if "accuracy" in history else None
    val_acc_key = "val_accuracy" if "val_accuracy" in history else None

    color_acc = "#1f77b4"
    if acc_key:
        ax1.plot(epochs, history[acc_key], color=color_acc, marker="o", markersize=3,
                  label="Train accuracy (proxy for Fig. 7's fidelity)")
    if val_acc_key:
        ax1.plot(epochs, history[val_acc_key], color=color_acc, linestyle="--", marker="s", markersize=3,
                  label="Validation accuracy")
    ax1.set_xlabel("Epoch (paper's Fig. 7 x-axis is \"Episode\" for LSTM training)")
    ax1.set_ylabel("Prediction accuracy fidelity", color=color_acc)
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    color_loss = "#d62728"
    ax2.plot(epochs, history["loss"], color=color_loss, marker="o", markersize=3, label="Train loss")
    if "val_loss" in history:
        ax2.plot(epochs, history["val_loss"], color=color_loss, linestyle="--", marker="s", markersize=3,
                  label="Validation loss")
    ax2.set_ylabel("Loss", color=color_loss)
    ax2.tick_params(axis="y", labelcolor=color_loss)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8)
    plt.title("LSTM attack predictor: fidelity & loss vs. training epoch\n(cf. paper Fig. 7)")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "fig7_lstm_fidelity_loss.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[ok] wrote {out_path}")
    return True


def fig8_confusion_matrix(results_dir: str, out_dir: str) -> bool:
    path = os.path.join(results_dir, "lstm_confusion.npz")
    if not os.path.exists(path):
        print(f"[skip] {path} not found -- run --mode train_lstm first.")
        return False

    data = np.load(path, allow_pickle=True)
    y_true, y_pred = data["y_true"], data["y_pred"]
    class_names = [str(c) for c in data["class_names"]]
    num_classes = len(class_names)

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Reality")
    ax.set_title(f"LSTM confusion matrix (cf. paper Fig. 8)\nFidelity={float(data['fidelity']):.4f}")

    for i in range(num_classes):
        for j in range(num_classes):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig8_confusion_matrix.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[ok] wrote {out_path}")

    # Table II equivalent: precision/recall/F1 per class, saved alongside the figure.
    report = {}
    for c, name in enumerate(class_names):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        report[name] = {"precision": precision, "recall": recall, "f1": f1, "support": int(cm[c, :].sum())}
    report_path = os.path.join(out_dir, "table2_classification_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[ok] wrote {report_path} (paper Table II equivalent)")
    return True


def fig9_dsr_over_training(results_dir: str, out_dir: str) -> bool:
    path = os.path.join(results_dir, "rl_training_curves.npz")
    if not os.path.exists(path):
        print(f"[skip] {path} not found -- run --mode train_rl first.")
        return False

    data = np.load(path)
    episodes, dsr = data["dsr_episodes"], data["dsr_values"]
    if len(episodes) == 0:
        print("[skip] no DSR windows recorded yet (increase num_episodes or lower eval_every_episodes).")
        return False

    plt.figure(figsize=(7, 4.5))
    plt.plot(episodes, dsr * 100, color="#d62728", marker="o", markersize=3, label="CM-MTD (this run)")
    plt.xlabel("Episode")
    plt.ylabel("Defense Success Ratio (%)")
    plt.ylim(0, 105)
    plt.title("Defense Success Ratio over training (cf. paper Fig. 9)")
    plt.figtext(
        0.5, -0.05,
        "Note: paper Fig. 9 compares CM-MTD against RRT+FRVM and DQ-RM+FRVM baselines\n"
        "(neither is implemented in this project) and averages 5 runs with error shadows;\n"
        "this plot is a single run of CM-MTD only.",
        ha="center", fontsize=8, style="italic",
    )
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig9_dsr_over_training.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ok] wrote {out_path}  (see in-figure note on baseline/run-count differences from the paper)")
    return True


def fig11_convergence(results_dir: str, out_dir: str, smoothing_window: int = 20) -> bool:
    path = os.path.join(results_dir, "rl_training_curves.npz")
    if not os.path.exists(path):
        print(f"[skip] {path} not found -- run --mode train_rl first.")
        return False

    data = np.load(path)
    upper, lower = data["upper_rewards"], data["lower_rewards"]
    if len(upper) == 0:
        print("[skip] no training episodes recorded.")
        return False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    episodes = np.arange(1, len(upper) + 1)
    ax1.plot(episodes, upper, color="#1f77b4", alpha=0.25, linewidth=1, label="raw")
    if len(upper) >= smoothing_window:
        smoothed = rolling_mean(upper, smoothing_window)
        ax1.plot(episodes[smoothing_window - 1:], smoothed, color="#1f77b4", linewidth=2,
                  label=f"{smoothing_window}-episode rolling mean")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward (upper layer / DQN)")
    ax1.set_title("Upper-layer convergence (cf. Fig. 11(a)/(c)/(e))")
    ax1.legend(fontsize=8)

    steps = np.arange(1, len(lower) + 1)
    ax2.plot(steps, lower, color="#d62728", alpha=0.25, linewidth=1, label="raw")
    if len(lower) >= smoothing_window:
        smoothed = rolling_mean(lower, smoothing_window)
        ax2.plot(steps[smoothing_window - 1:], smoothed, color="#d62728", linewidth=2,
                  label=f"{smoothing_window}-episode rolling mean")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Reward (lower layer / PPO)")
    ax2.set_title("Lower-layer convergence (cf. Fig. 11(b)/(d)/(f))")
    ax2.legend(fontsize=8)

    plt.figtext(0.5, -0.02, "Single run, no error shadow (paper averages 5 runs).",
                ha="center", fontsize=8, style="italic")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig11_convergence.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ok] wrote {out_path}")
    return True


def note_fig10_not_available() -> None:
    print(
        "\n[not generated] Fig. 10 (RTT / packet-loss network performance) needs real "
        "packet-level measurement (the paper uses iPerf against a live Mininet-WiFi "
        "topology). This project's environment.py is a probabilistic reward simulation, "
        "not a packet simulator -- it has no RTT/PLR signal to plot. Fabricating "
        "plausible-looking numbers here would misrepresent the experiment, so this figure "
        "is skipped until the real Mininet-WiFi/os-ken integration (docs/SETUP.md Part 2, "
        "README's 'Next steps') is in place.\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--smoothing-window", type=int, default=20,
                         help="Rolling-mean window for the convergence plot (episodes).")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results_dir = cfg["experiment"]["results_dir"]
    out_dir = os.path.join(results_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    any_ok = False
    any_ok |= fig7_lstm_fidelity_loss(results_dir, out_dir)
    any_ok |= fig8_confusion_matrix(results_dir, out_dir)
    any_ok |= fig9_dsr_over_training(results_dir, out_dir)
    any_ok |= fig11_convergence(results_dir, out_dir, smoothing_window=args.smoothing_window)
    note_fig10_not_available()

    if not any_ok:
        print("Nothing to plot yet -- run `python3 -m src.main --config ... --mode all` first.")
    else:
        print(f"Figures written to {out_dir}/")


if __name__ == "__main__":
    main()
