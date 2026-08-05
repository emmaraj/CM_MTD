"""
generate_dummy_data.py
-----------------------
Generates small placeholder .npy files with the SAME shapes/dtypes/paths
that config.yaml expects, purely so the pipeline can be smoke-tested
(imports, shapes, one training step) without needing the full
CICIDS-2017 CSVs on disk.

This is a development/testing utility only -- it uses np.random because
it is NOT part of the environment's stepping logic (see environment.py's
module docstring for why the actual RL environment never uses np.random).
Do not use this data to draw any conclusions about CM-MTD's performance;
run cicids2017.ipynb on the real dataset for that.

Usage:
    python scripts/generate_dummy_data.py --config config/config.yaml
"""

import argparse
import os
import sys

import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--n-features", type=int, default=40)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    rng = np.random.RandomState(0)
    num_classes = len(cfg["data"]["class_names"])

    # Mimic CICIDS-2017's severe class imbalance (Infiltration is rare)
    # so the class-weighting / capping logic actually gets exercised.
    class_probs = np.array([0.85, 0.149, 0.001])[:num_classes]
    class_probs = class_probs / class_probs.sum()

    def make_split(n):
        X = rng.randn(n, args.n_features).astype(np.float32)
        y = rng.choice(num_classes, size=n, p=class_probs).astype(np.int64)
        return X, y

    X_train, y_train = make_split(args.n_train)
    X_test, y_test = make_split(args.n_test)

    for key in ["x_train_path", "x_test_path", "y_train_path", "y_test_path"]:
        os.makedirs(os.path.dirname(cfg["data"][key]) or ".", exist_ok=True)

    np.save(cfg["data"]["x_train_path"], X_train)
    np.save(cfg["data"]["x_test_path"], X_test)
    np.save(cfg["data"]["y_train_path"], y_train)
    np.save(cfg["data"]["y_test_path"], y_test)

    print(f"Wrote dummy data: X_train{X_train.shape}, X_test{X_test.shape}, "
          f"y_train{y_train.shape}, y_test{y_test.shape}")
    print(f"Class distribution (train): {np.bincount(y_train, minlength=num_classes)}")


if __name__ == "__main__":
    main()
