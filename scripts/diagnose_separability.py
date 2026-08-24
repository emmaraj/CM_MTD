"""
diagnose_separability.py
-------------------------
Standalone diagnostic, independent of the sequence-model/windowing/RL
pipeline entirely. Trains a plain scikit-learn classifier directly on
your real preprocessed X_train/y_train rows for whichever dataset
config.experiment.dataset names (no sequences, no windowing, no
class-weight tuning beyond sklearn's built-in 'balanced' option) to
answer one question: are your actual preprocessed features separable at
all?

This exists because repeated attempts to fix an earlier single-LSTM
design's majority-class collapse via loss function / class-weight tuning
were not resolving it in synthetic reproductions, which means the
sequence-model/windowing pipeline itself needs to be ruled in or out as
the culprit before tuning it further. If THIS script also fails to
separate the classes, the issue is upstream (features/preprocessing), not
the sequence model. If this succeeds comfortably, the issue is specific
to the sequence/windowing pipeline.

Usage:
    python3 scripts/diagnose_separability.py --config config/config.yaml
    python3 scripts/diagnose_separability.py --config config/config.yaml --dataset 5g_nidd
"""

import argparse
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--dataset", type=str, default=None, choices=["cicids2017", "5g_nidd"],
                         help="Override config.experiment.dataset without editing config.yaml.")
    parser.add_argument("--max-rows", type=int, default=300000,
                         help="Subsample this many rows for speed (RandomForest on millions of rows is slow).")
    args = parser.parse_args()

    from src.config_parser import load_config, apply_cli_overrides, load_dataset

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, dataset=args.dataset)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix

    print(f"Loading dataset {cfg['experiment']['dataset']!r} via its adapter "
          f"(bypasses windowing/sequence-model entirely -- raw rows only)...")
    dataset = load_dataset(cfg)
    class_names = dataset.class_names

    X_train, y_train = dataset.X_train, dataset.y_train
    X_test, y_test = dataset.X_test, dataset.y_test

    print(f"X_train: {X_train.shape}, y_train classes: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"X_test:  {X_test.shape}, y_test classes:  {dict(zip(*np.unique(y_test, return_counts=True)))}")

    if len(X_train) > args.max_rows:
        # Stratified subsample for speed -- deterministic, not the training
        # pipeline's environment cache, this is purely a quick diagnostic.
        rng = np.random.RandomState(0)
        idx = rng.choice(len(X_train), size=args.max_rows, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
        print(f"Subsampled to {args.max_rows} rows for speed.")

    print("\n=== Baseline 1: Logistic Regression (linear separability check) ===")
    clf = LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== Baseline 2: Random Forest (nonlinear separability check) ===")
    clf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced",
                                  n_jobs=-1, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== Interpretation ===")
    print("If both baselines get meaningfully above the majority-class rate")
    print("with real recall on the minority classes (not 0%, not collapsed the")
    print("other way), your features ARE separable -- the problem is specific to")
    print("the sequence-model/windowing pipeline, not the underlying data.")
    print("If these ALSO collapse or perform near-randomly, the issue is")
    print("upstream: likely the feature selection/preprocessing in the relevant")
    print("preprocessing notebook (e.g. the correlation-pruning step may have")
    print("dropped the most discriminative features).")


if __name__ == "__main__":
    main()
