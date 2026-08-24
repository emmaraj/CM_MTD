"""
compare_architectures.py
-------------------------
Trains BOTH the LSTM and Transformer Stage-2 predictors on identical data
(same Stage-1 classifier, same train/test split, same class weights, same
random seed) and produces a side-by-side comparison across predictive
performance, fairness, trust, and energy -- the evidence base for the
methodology report's architecture-comparison section.

Runs against whichever dataset config.experiment.dataset names (override
with --dataset) -- experiment.predictor is ignored here since the whole
point of this script is to run both predictors in the same pass.

Usage:
    python3 scripts/compare_architectures.py --config config/config.yaml
    python3 scripts/compare_architectures.py --config config/config.yaml --dataset 5g_nidd
    python3 scripts/compare_architectures.py --config config/config.yaml --output results/comparison.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_one(cfg: dict, dataset, predictor_type: str, logger):
    from src.main import _build_predictor
    from src.config_parser import compute_class_weights, reset_tf_session
    from src.models import EventClassifier
    from src import metrics as metrics_mod

    reset_tf_session()
    stage2_cfg = cfg[predictor_type]

    clf_cfg = cfg["event_classifier"]
    classifier = EventClassifier(n_estimators=clf_cfg["n_estimators"], max_depth=clf_cfg["max_depth"],
                                  seed=cfg["experiment"]["seed"])
    classifier.fit(dataset.X_train, dataset.y_train)
    train_pred = classifier.predict(dataset.X_train)
    test_pred = classifier.predict(dataset.X_test)

    class_weight = compute_class_weights(
        dataset.y_train, dataset.num_classes,
        strategy=stage2_cfg["class_weight_strategy"], max_ratio=stage2_cfg["max_class_weight_ratio"],
    )

    with metrics_mod.track_energy(f"compare_{predictor_type}") as energy_report:
        predictor = _build_predictor(predictor_type, dataset.num_classes, cfg)
        predictor.fit(train_pred, dataset.y_train, class_weight=class_weight, seed=cfg["experiment"]["seed"])

    fidelity_metrics = predictor.compute_fidelity(test_pred, dataset.y_test)
    fairness = metrics_mod.class_fairness(fidelity_metrics["per_class_accuracy"])

    windows, targets = predictor.build_sliding_windows(test_pred, dataset.y_test)
    probs = predictor.predict_proba(windows)
    ece = metrics_mod.expected_calibration_error(probs, targets)
    stability = metrics_mod.prediction_stability(predictor, test_pred, dataset.y_test, seed=cfg["experiment"]["seed"])

    result = {
        "predictor_type": predictor_type,
        "fidelity": fidelity_metrics["fidelity"],
        "per_class_accuracy": fidelity_metrics["per_class_accuracy"],
        "fairness_recall_gap": fairness["recall_gap"],
        "fairness_equity_ratio": fairness["equity_ratio"],
        "calibration_ece": ece["ece"],
        "prediction_flip_rate": stability["flip_rate"],
        "training_wall_clock_seconds": energy_report.wall_clock_seconds,
        "training_estimated_kwh": energy_report.estimated_energy_kwh,
        "energy_measurement_method": energy_report.method,
        "trainable_params": metrics_mod.count_trainable_params(predictor.model),
    }

    sample = windows[:1]
    result["inference_ms_per_call"] = metrics_mod.benchmark_inference_latency(predictor.predict_next_events, sample)

    if predictor_type == "transformer":
        attn = predictor.get_attention_weights(windows[:min(2000, len(windows))])
        result["attention_entropy"] = metrics_mod.attention_entropy(attn)["mean_normalized_entropy"]

    logger.info("%s: fidelity=%.4f fairness_gap=%.4f ECE=%.4f train_time=%.1fs params=%d",
                predictor_type, result["fidelity"], result["fairness_recall_gap"],
                result["calibration_ece"], result["training_wall_clock_seconds"], result["trainable_params"])
    return result


def print_comparison_table(results: dict) -> None:
    lstm, transformer = results.get("lstm"), results.get("transformer")
    if not (lstm and transformer):
        return

    rows = [
        ("Fidelity (Eq. 18)", "fidelity", "{:.4f}"),
        ("Fairness: recall gap (lower better)", "fairness_recall_gap", "{:.4f}"),
        ("Fairness: equity ratio (higher better)", "fairness_equity_ratio", "{:.4f}"),
        ("Trust: ECE (lower better)", "calibration_ece", "{:.4f}"),
        ("Trust: prediction flip rate", "prediction_flip_rate", "{:.4f}"),
        ("Energy: training time (s)", "training_wall_clock_seconds", "{:.1f}"),
        ("Energy: estimated kWh", "training_estimated_kwh", "{:.6f}"),
        ("Energy: inference (ms/call)", "inference_ms_per_call", "{:.3f}"),
        ("Model size (params)", "trainable_params", "{:,}"),
    ]
    print(f"\n{'Metric':<42}{'LSTM':>15}{'Transformer':>15}")
    print("-" * 72)
    for label, key, fmt in rows:
        lv, tv = lstm.get(key), transformer.get(key)
        lv_str = fmt.format(lv) if lv is not None else "N/A"
        tv_str = fmt.format(tv) if tv is not None else "N/A"
        print(f"{label:<42}{lv_str:>15}{tv_str:>15}")
    if "attention_entropy" in transformer:
        print(f"{'Trust: attention entropy (Transformer only)':<42}{'—':>15}{transformer['attention_entropy']:>15.4f}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--dataset", type=str, default=None, choices=["cicids2017", "5g_nidd"],
                         help="Override config.experiment.dataset without editing config.yaml.")
    parser.add_argument("--output", type=str, default=None,
                         help="Where to save the JSON comparison "
                              "(default: results/<dataset>/architecture_comparison.json)")
    args = parser.parse_args()

    from src.config_parser import load_dataset, apply_cli_overrides, set_global_seed, configure_device, setup_logging, load_config

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, dataset=args.dataset)
    dataset_name = cfg["experiment"]["dataset"]

    logger = setup_logging(cfg["experiment"]["log_dir"])
    set_global_seed(cfg["experiment"]["seed"])
    configure_device(cfg["experiment"]["device"])
    dataset = load_dataset(cfg)
    logger.info("Comparing LSTM vs Transformer on %s (input_dim=%d, num_classes=%d)",
                dataset_name, dataset.input_dim, dataset.num_classes)

    results = {
        "lstm": run_one(cfg, dataset, "lstm", logger),
        "transformer": run_one(cfg, dataset, "transformer", logger),
    }

    print_comparison_table(results)

    out_path = args.output or os.path.join(cfg["experiment"]["results_dir"], dataset_name,
                                            "architecture_comparison.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Saved comparison to {out_path}")


if __name__ == "__main__":
    main()
