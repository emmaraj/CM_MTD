"""
nidd5g.py
---------
Adapter for 5G-NIDD as preprocessed by notebooks/5g-nidd.ipynb, which
writes a self-contained directory of .npy arrays (plus a scaler and
preprocessing_metadata.json) under config.datasets.5g_nidd.data_dir
(default: dataset/5g-nidd/drl_dataset/, i.e. the notebook's own
OUTPUT_DIR).

Differences from the CICIDS-2017 adapter that make a shared loader the
wrong call (see src/datasets/base.py's module docstring for the full
rationale):
  * This notebook does NOT produce dones_{split}_seq.npy -- it has no
    timestamp column in the raw 5G-NIDD CSV to derive real episode
    boundaries from, and explicitly assumes raw row order is
    chronological rather than synthesizing gap flags. dones_train_seq is
    therefore always None for this dataset.
  * preprocessing_metadata.json's schema differs from CICIDS-2017's (see
    base.py) -- kept as opaque raw_metadata here, not parsed.
  * The notebook's label_mapping happens to produce the SAME three
    semantic classes as CICIDS-2017's current notebook (Benign /
    Scan/Infiltration / DDoS via SYNScan+TCPConnectScan+UDPScan+PortScan
    -> Scan/Infiltration, and ICMPFlood+UDPFlood+SYNFlood+HTTPFlood+
    SlowrateDoS -> DDoS) -- but that's declared explicitly below via
    RECONNAISSANCE_CLASS_NAME/FLOODING_CLASS_NAME rather than assumed
    identical to CICIDS-2017's just because the strings happen to match
    today. If a future revision of this notebook renames or re-splits
    these classes, only this one constant needs to change.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np

from .base import DatasetAdapter, DatasetBundle

logger = logging.getLogger("cm_mtd")


class Nidd5gAdapter(DatasetAdapter):
    name = "5g_nidd"

    RECONNAISSANCE_CLASS_NAME = "Scan/Infiltration"
    FLOODING_CLASS_NAME = "DDoS"

    def load(self, dataset_cfg: dict) -> DatasetBundle:
        data_dir = dataset_cfg["data_dir"]
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(
                f"5G-NIDD data_dir not found: '{data_dir}' (config.datasets.5g_nidd.data_dir). "
                f"Run notebooks/5g-nidd.ipynb through its 'Save .npy Files' / 'Save Metadata' "
                f"cells first, or point config.yaml at wherever the notebook's OUTPUT_DIR "
                f"actually landed."
            )

        def _load(fname: str, required: bool = True):
            path = os.path.join(data_dir, fname)
            if not os.path.exists(path):
                if required:
                    raise FileNotFoundError(
                        f"Expected '{fname}' under 5G-NIDD data_dir '{data_dir}' but it's not "
                        f"there. This usually means the preprocessing notebook was interrupted "
                        f"before its save cells, or data_dir points at an older/partial run."
                    )
                return None
            arr = np.load(path, allow_pickle=False)
            logger.info("Loaded %s -> shape %s dtype %s", path, arr.shape, arr.dtype)
            return arr

        X_train = _load("X_train.npy").astype(np.float32)
        y_train = _load("y_train.npy").astype(np.int64).reshape(-1)
        X_test = _load("X_test.npy").astype(np.float32)
        y_test = _load("y_test.npy").astype(np.int64).reshape(-1)

        X_val = _load("X_val.npy", required=False)
        y_val = _load("y_val.npy", required=False)
        X_val = X_val.astype(np.float32) if X_val is not None else None
        y_val = y_val.astype(np.int64).reshape(-1) if y_val is not None else None

        class_names_arr = _load("class_names.npy", required=False)
        class_names = list(class_names_arr) if class_names_arr is not None else list(
            dataset_cfg.get("class_names", [])
        )
        class_names = [str(c) for c in class_names]

        feature_names_arr = _load("feature_names.npy", required=False)
        feature_names = [str(c) for c in feature_names_arr] if feature_names_arr is not None else []

        # No dones_train_seq for this dataset -- see module docstring.
        X_train_seq = _load("X_train_seq.npy", required=False)
        y_train_seq = _load("y_train_seq.npy", required=False)

        metadata_path = os.path.join(data_dir, "preprocessing_metadata.json")
        raw_metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                raw_metadata = json.load(f)
        else:
            logger.warning(
                "No preprocessing_metadata.json found under '%s' -- proceeding without it "
                "(only used for logging/provenance, not required for training).",
                data_dir,
            )

        return DatasetBundle(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            X_val=X_val, y_val=y_val,
            class_names=class_names,
            feature_names=feature_names,
            X_train_seq=X_train_seq, y_train_seq=y_train_seq, dones_train_seq=None,
            reconnaissance_class_name=self.RECONNAISSANCE_CLASS_NAME,
            flooding_class_name=self.FLOODING_CLASS_NAME,
            raw_metadata=raw_metadata,
        )
