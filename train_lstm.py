"""
train_lstm.py — Train the LSTM attack prediction model.

Usage:
    python train_lstm.py [--config config/config.yaml] [--seed 42]
                         [--data-dir data/cicids2017] [--synthetic]

Steps:
  1. Load CICIDS2017 (or generate synthetic data)
  2. Preprocess and build event sequences
  3. Train LSTMNet (Embedding → LSTM → Dense → Softmax)
  4. Evaluate: fidelity, confusion matrix, per-class P/R/F1
  5. Save model + metrics + generate Figures 7 and 8
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# ── Ensure project root is on PYTHONPATH ─────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from utils.seed_utils import set_global_seed
from utils.logger import setup_logger
from datasets.cicids2017_loader import ATTACK_CLASS_MAP, CLASS_NAMES, N_CLASSES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LSTM Attack Predictor")
    p.add_argument("--config",   default="config/config.yaml", help="Config YAML path")
    p.add_argument("--seed",     type=int, default=42,          help="Random seed")
    p.add_argument("--data-dir", default="data/cicids2017",     help="CICIDS2017 CSV directory")
    p.add_argument("--synthetic",action="store_true",           help="Use synthetic data")
    p.add_argument("--epochs",   type=int, default=None,        help="Override max epochs")
    p.add_argument("--out-dir",  default="results",             help="Output directory")
    p.add_argument("--no-figures", action="store_true",         help="Skip figure generation")
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    # ── Setup ─────────────────────────────────────────────────────────────────
    logger = setup_logger("lstm_train", log_dir="logs")
    config = load_config(args.config)
    set_global_seed(args.seed)

    lstm_cfg = config["lstm"]
    data_cfg = config["data"]
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("CM-MTD: LSTM Attack Prediction Training")
    logger.info(f"  Seed: {args.seed} | Synthetic: {args.synthetic}")
    logger.info("=" * 60)

    # ── Data Loading ──────────────────────────────────────────────────────────
    seq_len  = data_cfg.get("sequence_length", 10)
    n_classes = N_CLASSES

    if args.synthetic:
        logger.info("Using synthetic data (CICIDS2017 not found or --synthetic flag set)")
        from datasets.sequence_builder import SyntheticDataGenerator
        gen = SyntheticDataGenerator(n_classes=n_classes, seed=args.seed)
        X_all, y_all = gen.generate_event_sequence(
            n_nodes=config["network"]["n_nodes"],
            n_steps=50_000,
            sequence_length=seq_len,
        )
        scenario_key = "dos_scan"
        X_cicids, y_cicids = gen.generate_event_sequence(
            n_nodes=config["network"]["n_nodes"],
            n_steps=30_000,
            sequence_length=seq_len,
        )
    else:
        try:
            from datasets.cicids2017_loader import CICIDS2017Loader
            from datasets.preprocessor import CICIDS2017Preprocessor
            from datasets.sequence_builder import SecurityEventSequenceBuilder

            logger.info(f"Loading CICIDS2017 from {args.data_dir}")
            loader = CICIDS2017Loader(data_dir=args.data_dir)
            df_raw = loader.load_all()

            preprocessor = CICIDS2017Preprocessor(config=config)
            X_feat, y_labels, feat_names = preprocessor.fit_transform(df_raw)

            # Balance dataset
            X_feat, y_labels = preprocessor.balance_undersample(
                X_feat, y_labels,
                max_per_class=data_cfg.get("max_samples_per_class", 50000),
                seed=args.seed,
            )

            # Build event sequences
            builder = SecurityEventSequenceBuilder(
                sequence_length=seq_len, n_classes=n_classes
            )
            X_all, y_all = builder.build_from_labels(y_labels)
            scenario_key = "cicids2017"
            X_cicids, y_cicids = X_all, y_all
            logger.info(f"Built {len(X_all):,} sequences from CICIDS2017")

        except (FileNotFoundError, Exception) as e:
            logger.warning(f"CICIDS2017 load failed ({e}). Falling back to synthetic data.")
            args.synthetic = True
            return run(args)  # Retry with synthetic

    # ── Train/Val/Test Split ──────────────────────────────────────────────────
    from datasets.sequence_builder import SecurityEventSequenceBuilder
    builder = SecurityEventSequenceBuilder(sequence_length=seq_len, n_classes=n_classes)
    splits = builder.train_val_test_split(
        X_all, y_all,
        train_frac=data_cfg.get("train_split", 0.70),
        val_frac=data_cfg.get("val_split", 0.10),
        test_frac=data_cfg.get("test_split", 0.20),
        stratify=True,
        seed=args.seed,
    )
    X_train, y_train = splits["train"]
    X_val,   y_val   = splits["val"]
    X_test,  y_test  = splits["test"]

    logger.info(
        f"Splits: train={len(X_train):,} | val={len(X_val):,} | test={len(X_test):,}"
    )

    # ── Build Model ───────────────────────────────────────────────────────────
    from models.lstm_predictor import LSTMAttackPredictor
    model = LSTMAttackPredictor.from_config(config)
    model.build()

    # ── Train ─────────────────────────────────────────────────────────────────
    epochs = args.epochs or lstm_cfg.get("epochs", 100)
    history = model.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=epochs,
        batch_size=lstm_cfg.get("batch_size", 256),
        patience=lstm_cfg.get("early_stopping_patience", 10),
        lr_patience=lstm_cfg.get("lr_reduce_patience", 5),
        lr_factor=lstm_cfg.get("lr_reduce_factor", 0.5),
        checkpoint_path="checkpoints/lstm_best.keras",
        log_dir="logs",
        verbose=1,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    logger.info("Evaluating on test set...")
    metrics = model.evaluate(X_test, y_test, class_names=CLASS_NAMES)

    # Per-class table
    logger.info("\n── Per-Class Metrics ──────────────────────────────────────")
    logger.info(f"{'Class':<15} {'Accuracy':>10} {'Recall':>8} {'F1':>8}")
    for i, cls in enumerate(CLASS_NAMES):
        pc = metrics["per_class"]
        logger.info(
            f"  {cls:<13} {pc['precision'][i]*100:>9.2f}%  "
            f"{pc['recall'][i]*100:>7.2f}%  {pc['f1'][i]*100:>7.2f}%"
        )
    logger.info(f"\n  Overall Fidelity:  {metrics['fidelity']*100:.2f}%")
    logger.info(f"  Overall F1:        {metrics['f1']*100:.2f}%")

    # ── Save results ──────────────────────────────────────────────────────────
    model.save("checkpoints/lstm_final.keras")

    metrics_out = {
        "seed":     args.seed,
        "scenario": scenario_key,
        **metrics,
    }
    out_path = Path(args.out_dir) / "metrics" / "lstm_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)
    logger.info(f"Metrics saved to {out_path}")

    # ── Generate Figures 7 & 8 ────────────────────────────────────────────────
    if not args.no_figures:
        logger.info("Generating Figures 7 and 8...")
        from visualizations.figure_generator import FigureGenerator
        fgen = FigureGenerator(out_dir=f"{args.out_dir}/figures")

        # Figure 7: accuracy + loss curves
        fig7_paths = fgen.figure7_prediction_curves(
            histories={scenario_key: history}
        )
        logger.info(f"Fig 7 saved: {fig7_paths}")

        # Figure 8: confusion matrices
        import numpy as np
        cm = np.array(metrics["confusion_matrix"])
        fig8_paths = fgen.figure8_confusion_matrices(
            cms={scenario_key: cm},
            class_names=CLASS_NAMES,
        )
        logger.info(f"Fig 8 saved: {fig8_paths}")

    logger.info("LSTM training complete.")
    return model, metrics


if __name__ == "__main__":
    args = parse_args()
    run(args)
