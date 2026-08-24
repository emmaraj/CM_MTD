"""
generate_dummy_data.py
-----------------------
Generates small placeholder .npy directories with the SAME on-disk
contract each real preprocessing notebook produces (notebooks/
cicids2017.ipynb, notebooks/5g-nidd.ipynb) -- filenames, shapes, dtypes,
directory layout, feature_names.npy/class_names.npy/
preprocessing_metadata.json -- purely so the pipeline can be smoke-tested
(imports, adapter loading, shapes, one training step) without needing the
full real datasets on disk.

This is a development/testing utility only -- it uses np.random because
it is NOT part of the environment's stepping logic (see environment.py's
module docstring for why the actual RL environment never uses np.random).
Do not use this data to draw any conclusions about CM-MTD's performance;
run the real preprocessing notebooks on real data for that.

Per-dataset class imbalance is deliberately matched to what each real
notebook actually produced (see each dataset's block below) so the
class-weighting / capping logic in config_parser.compute_class_weights
gets exercised the same way it would on real data -- CICIDS-2017 is
heavily Benign-skewed, 5G-NIDD is comparatively balanced with DDoS as the
plurality class, and a fixture that used the same ratio for both would
miss that difference entirely.

Usage:
    python scripts/generate_dummy_data.py --config config/config.yaml
    python scripts/generate_dummy_data.py --config config/config.yaml --dataset 5g_nidd
    python scripts/generate_dummy_data.py --config config/config.yaml --dataset all
"""

import argparse
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Real class distributions observed from each notebook's own printed
# training-split summary, used only to make the dummy fixture exercise
# the same imbalance the real data would. Order matches class_names
# below: [Benign, Scan/Infiltration, DDoS].
_REAL_TRAIN_CLASS_PROBS = {
    "cicids2017": [0.8385, 0.0781, 0.0834],   # heavily Benign-skewed
    "5g_nidd": [0.4407, 0.0576, 0.5017],      # DDoS-plurality, comparatively balanced
}

_CLASS_NAMES = ["Benign", "Scan/Infiltration", "DDoS"]


def _make_split(rng: np.random.RandomState, n: int, n_features: int, class_probs: np.ndarray):
    X = rng.rand(n, n_features).astype(np.float32)  # MinMaxScaler-style [0,1] features
    y = rng.choice(len(class_probs), size=n, p=class_probs).astype(np.int64)
    return X, y


def _make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int, stride: int = 1):
    n = len(X) - seq_len + 1
    if n <= 0:
        return None, None
    X_seq = np.stack([X[i:i + seq_len] for i in range(0, n, stride)], axis=0).astype(np.float32)
    y_seq = np.array([y[i + seq_len - 1] for i in range(0, n, stride)], dtype=np.int64)
    return X_seq, y_seq


def _write_common_arrays(out_dir: str, n_features: int, n_train: int, n_val: int, n_test: int,
                          seq_len: int, class_probs, seed: int, with_dones: bool):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    class_probs = np.array(class_probs, dtype=np.float64)
    class_probs = class_probs / class_probs.sum()

    X_train, y_train = _make_split(rng, n_train, n_features, class_probs)
    X_val, y_val = _make_split(rng, n_val, n_features, class_probs)
    X_test, y_test = _make_split(rng, n_test, n_features, class_probs)

    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "X_val.npy"), X_val)
    np.save(os.path.join(out_dir, "y_val.npy"), y_val)
    np.save(os.path.join(out_dir, "X_test.npy"), X_test)
    np.save(os.path.join(out_dir, "y_test.npy"), y_test)

    X_train_seq, y_train_seq = _make_sequences(X_train, y_train, seq_len)
    if X_train_seq is not None:
        np.save(os.path.join(out_dir, "X_train_seq.npy"), X_train_seq)
        np.save(os.path.join(out_dir, "y_train_seq.npy"), y_train_seq)
        if with_dones:
            # Real dones flags mark genuine time-gap episode boundaries;
            # the dummy fixture has no real timestamps to derive that
            # from, so it's all-zero (matches what the real CICIDS-2017
            # notebook itself produces when its Timestamp column is
            # absent and it falls back to an integer index -- see that
            # notebook's "Temporal Ordering" cell).
            dones_train_seq = np.zeros(len(y_train_seq), dtype=np.int8)
            np.save(os.path.join(out_dir, "dones_train_seq.npy"), dones_train_seq)

        for split_name, X_s, y_s in [("val", X_val, y_val), ("test", X_test, y_test)]:
            X_seq, y_seq = _make_sequences(X_s, y_s, seq_len)
            if X_seq is not None:
                np.save(os.path.join(out_dir, f"X_{split_name}_seq.npy"), X_seq)
                np.save(os.path.join(out_dir, f"y_{split_name}_seq.npy"), y_seq)
                if with_dones:
                    np.save(os.path.join(out_dir, f"dones_{split_name}_seq.npy"),
                            np.zeros(len(y_seq), dtype=np.int8))

    feature_names = np.array([f"feature_{i}" for i in range(n_features)])
    np.save(os.path.join(out_dir, "feature_names.npy"), feature_names)
    np.save(os.path.join(out_dir, "class_names.npy"), np.array(_CLASS_NAMES))

    print(f"  X_train{X_train.shape} X_val{X_val.shape} X_test{X_test.shape}")
    print(f"  class distribution (train): {np.bincount(y_train, minlength=len(_CLASS_NAMES))}")
    return X_train, y_train


def generate_cicids2017(dataset_cfg: dict, n_train: int, n_val: int, n_test: int,
                         n_features: int, seq_len: int, seed: int) -> None:
    out_dir = dataset_cfg["data_dir"]
    print(f"Generating CICIDS-2017-shaped dummy data -> {out_dir}")
    X_train, y_train = _write_common_arrays(
        out_dir, n_features, n_train, n_val, n_test, seq_len,
        _REAL_TRAIN_CLASS_PROBS["cicids2017"], seed, with_dones=True,
    )

    metadata = {
        "dataset_name": "CICIDS-2017 (DUMMY FIXTURE -- generated by scripts/generate_dummy_data.py)",
        "dataset_dir": "<dummy>",
        "class_mapping": {"<dummy>": "see notebooks/cicids2017.ipynb for the real mapping"},
        "class_to_id": {name: i for i, name in enumerate(_CLASS_NAMES)},
        "num_original_features": n_features,
        "num_final_features": n_features,
        "final_features": [f"feature_{i}" for i in range(n_features)],
        "correlation_threshold": 0.90,
        "scaler_type": "minmax",
        "sequence_length": seq_len,
        "sequence_stride": 1,
        "sequence_label_policy": "last",
        "train_ratio": 0.80, "val_ratio": 0.10, "test_ratio": 0.10,
        "train_samples": n_train, "val_samples": n_val, "test_samples": n_test,
        "train_sequences": max(0, n_train - seq_len + 1),
    }
    with open(os.path.join(out_dir, "preprocessing_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)


def generate_5g_nidd(dataset_cfg: dict, n_train: int, n_val: int, n_test: int,
                      n_features: int, seq_len: int, seed: int) -> None:
    out_dir = dataset_cfg["data_dir"]
    print(f"Generating 5G-NIDD-shaped dummy data -> {out_dir}")
    X_train, y_train = _write_common_arrays(
        out_dir, n_features, n_train, n_val, n_test, seq_len,
        _REAL_TRAIN_CLASS_PROBS["5g_nidd"], seed, with_dones=False,
    )

    metadata = {
        "dataset": "5G-NIDD (DUMMY FIXTURE -- generated by scripts/generate_dummy_data.py)",
        "preprocessing_logic": "Chronological split, leak-free scaled, for DRL states",
        "original_samples": n_train + n_val + n_test,
        "cleaned_samples": n_train + n_val + n_test,
        "features_before_selection": n_features,
        "features_after_selection": n_features,
        "train_samples": n_train, "val_samples": n_val, "test_samples": n_test,
        "scaler": "MinMaxScaler",
        "correlation_threshold": 0.90,
        "use_resampling": False,
        "sequence_mode": True,
        "sequence_length": seq_len,
        "stride": 1,
        "random_seed": seed,
        "class_mapping": {str(i): name for i, name in enumerate(_CLASS_NAMES)},
    }
    with open(os.path.join(out_dir, "preprocessing_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)


_GENERATORS = {
    "cicids2017": generate_cicids2017,
    "5g_nidd": generate_5g_nidd,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--dataset", type=str, default=None, choices=["cicids2017", "5g_nidd", "all"],
                         help="Which dataset to generate a fixture for. Defaults to "
                              "config.experiment.dataset. Use 'all' to generate both.")
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--n-features", type=int, default=None,
                         help="Defaults to each dataset's real post-selection feature count "
                              "(37 for CICIDS-2017, 28 for 5G-NIDD) if not given.")
    parser.add_argument("--seq-len", type=int, default=10)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("experiment", {}).get("seed", 42)
    target = args.dataset or cfg["experiment"]["dataset"]
    targets = list(_GENERATORS.keys()) if target == "all" else [target]

    _default_n_features = {"cicids2017": 37, "5g_nidd": 28}

    for name in targets:
        dataset_cfg = cfg["datasets"][name]
        n_features = args.n_features or _default_n_features[name]
        _GENERATORS[name](dataset_cfg, args.n_train, args.n_val, args.n_test, n_features, args.seq_len, seed)
        print(f"Done: {name}\n")


if __name__ == "__main__":
    main()
