"""
cicids2017.py
-------------
Adapter for CICIDS-2017 as preprocessed by the current
notebooks/cicids2017.ipynb, which writes a self-contained directory of
.npy arrays (plus a scaler/variance-selector and preprocessing_metadata.json)
under config.datasets.cicids2017.data_dir
(default: dataset/cicids2017/drl_dataset/, i.e. the notebook's own
CONFIG["OUTPUT_DIR"]).

This adapter is the ONLY place that knows CICIDS-2017-specific facts:
  * the exact filenames the notebook writes, and that val/seq/dones
    arrays are present
  * which of its three classes plays the "reconnaissance" vs "flooding"
    role for environment.py's HAM/RM defense logic -- fixed by the
    notebook's own CLASS_TO_ID = {"Benign": 0, "Scan/Infiltration": 1,
    "DDoS": 2}, not something class_names.npy alone can tell you (it's
    just an ordered list of strings)

Note the notebook's LABEL_MAPPING collapses PortScan, Infiltration, all
three Web Attack subtypes, FTP-Patator, SSH-Patator, and Bot into
"Scan/Infiltration", and all DoS/DDoS/Heartbleed variants into "DDoS" --
see preprocessing_metadata.json's own "class_mapping" field (raw label ->
mapped class name) for the authoritative record of that collapse, saved
by the notebook itself.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np

from .base import DatasetAdapter, DatasetBundle

logger = logging.getLogger("cm_mtd")


class Cicids2017Adapter(DatasetAdapter):
    name = "cicids2017"

    # Fixed by notebooks/cicids2017.ipynb's CLASS_TO_ID -- not inferred,
    # since class_names.npy alone is just an ordered list of strings and
    # doesn't say WHICH entry is which archetype. See DatasetBundle's
    # semantic-role fields and environment.py's module docstring.
    RECONNAISSANCE_CLASS_NAME = "Scan/Infiltration"
    FLOODING_CLASS_NAME = "DDoS"

    def load(self, dataset_cfg: dict) -> DatasetBundle:
        data_dir = dataset_cfg["data_dir"]
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(
                f"CICIDS-2017 data_dir not found: '{data_dir}' "
                f"(config.datasets.cicids2017.data_dir). Run "
                f"notebooks/cicids2017.ipynb through its 'Save .npy Arrays & "
                f"Save Metadata' cell first, or point config.yaml at wherever "
                f"the notebook's CONFIG['OUTPUT_DIR'] actually landed."
            )

        def _load(fname: str, required: bool = True):
            path = os.path.join(data_dir, fname)
            if not os.path.exists(path):
                if required:
                    raise FileNotFoundError(
                        f"Expected '{fname}' under CICIDS-2017 data_dir '{data_dir}' but it's "
                        f"not there. This usually means the preprocessing notebook was "
                        f"interrupted before its 'Save .npy Arrays & Save Metadata' cell, or "
                        f"data_dir points at an older/partial run."
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
        # np.load of a string array returns numpy str_ objects; normalize to
        # plain str so downstream `in` / `.index()` checks against literal
        # strings (e.g. reconnaissance_class_name) behave exactly as expected.
        class_names = [str(c) for c in class_names]

        feature_names_arr = _load("feature_names.npy", required=False)
        feature_names = [str(c) for c in feature_names_arr] if feature_names_arr is not None else []

        # Optional -- see DatasetBundle's docstring for why these aren't
        # required and aren't (yet) consumed downstream.
        X_train_seq = _load("X_train_seq.npy", required=False)
        y_train_seq = _load("y_train_seq.npy", required=False)
        dones_train_seq = _load("dones_train_seq.npy", required=False)

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
            X_train_seq=X_train_seq, y_train_seq=y_train_seq, dones_train_seq=dones_train_seq,
            reconnaissance_class_name=self.RECONNAISSANCE_CLASS_NAME,
            flooding_class_name=self.FLOODING_CLASS_NAME,
            raw_metadata=raw_metadata,
        )
