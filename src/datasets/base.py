"""
base.py
-------
Abstract dataset adapter + the DatasetBundle every adapter must produce.

Why an adapter per dataset (not one shared loader):

CICIDS-2017 and 5G-NIDD's preprocessing notebooks (notebooks/cicids2017.ipynb,
notebooks/5g-nidd.ipynb) agree on the important things -- both now emit a
self-contained directory of .npy arrays under a fixed, versioned contract
(X_train.npy/y_train.npy/X_val.npy/... , feature_names.npy, class_names.npy,
preprocessing_metadata.json), and -- as of the current notebook revisions --
both use the SAME 3-class taxonomy (Benign / Scan/Infiltration / DDoS). But
they differ in exactly the ways a rigid shared loader would silently get
wrong:

  * CICIDS-2017's notebook also emits per-sequence episode-boundary
    dones_{split}_seq.npy arrays (derived from real, or a synthetic
    fallback, per-row timestamp). 5G-NIDD's notebook does not produce
    these at all -- it has no timestamp column to derive them from and
    assumes raw CSV order is chronological.
  * preprocessing_metadata.json's own SCHEMA differs between the two:
    CICIDS-2017 keys the dataset name "dataset_name" and its
    "class_mapping" as {raw_label: mapped_class_name}; 5G-NIDD keys the
    dataset name "dataset" and its "class_mapping" as
    {str(class_id): class_name} -- same field name, opposite meaning.
    Application code must never branch on raw_metadata's keys because of
    this; it's kept only for logging/provenance (see DatasetBundle).
  * CICIDS-2017 also saves a variance_selector.pkl (joblib.dump);
    5G-NIDD does not save one at all. Neither is currently consumed
    downstream (see the module docstring in each adapter).

Rather than have config_parser.py (or worse, environment.py) branch on
dataset name inline -- the kind of thing that silently rots the next
time a preprocessing notebook changes shape -- each dataset gets its own
thin adapter that knows its own directory contract. Everything downstream
(config_parser.py, environment.py, models.py) only ever sees the common
DatasetBundle shape below, so adding a third dataset later is "write one
more adapter + register it," not "touch every consumer."
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class DatasetBundle:
    """
    The single, dataset-agnostic shape every adapter must produce.
    input_dim / num_classes are inferred from the arrays themselves at
    construction time -- never hardcoded here or anywhere downstream,
    per the project's dataset-agnostic requirement.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    class_names: list

    # Optional validation split. Both current notebooks produce one
    # (80/10/10 chronological split) -- optional here so a future
    # dataset/adapter without one still satisfies the contract, and so
    # config_parser.Dataset's pre-refactor callers that never expected a
    # val set don't need to change.
    X_val: Optional[np.ndarray] = None
    y_val: Optional[np.ndarray] = None

    # Optional, pre-built RAW-FEATURE sequence windows + episode-boundary
    # "done" flags, straight from the notebook's own sliding-window pass
    # (train split only -- val/test sequence arrays exist on disk too,
    # but nothing downstream currently consumes them, so they aren't
    # threaded through here to avoid holding unused memory).
    #
    # IMPORTANT: these are NOT currently consumed by Stage 2
    # (LSTMAttackPredictor / TransformerAttackPredictor build their OWN
    # sliding windows over Stage-1's classified LABEL sequences, not raw
    # features -- see models.py's build_sliding_windows). They're
    # exposed here purely so the notebook's output isn't silently
    # dropped on load; a natural next step is wiring dones_train_seq
    # into environment.py's row-cycling so episode boundaries reflect
    # genuine time gaps instead of the current artificial round-robin
    # (see README "Next steps"). dones_train_seq is None for datasets
    # (like 5G-NIDD) whose notebook doesn't produce it.
    X_train_seq: Optional[np.ndarray] = None
    y_train_seq: Optional[np.ndarray] = None
    dones_train_seq: Optional[np.ndarray] = None

    # Feature names, in column order matching X_*. Cosmetic (logging) --
    # nothing downstream indexes into X by name.
    feature_names: list = field(default_factory=list)

    # Semantic roles for environment.py's HAM/RM defense logic (see that
    # module's docstring): which entry of class_names is the
    # reconnaissance-type archetype HAM defends against, and which is
    # the flooding-type archetype RM defends against. Populated by the
    # ADAPTER, never hardcoded downstream -- this is what fixed the
    # class-name mismatch between what a preprocessing notebook happens
    # to call a class and what environment.py goes looking for. See the
    # `reconnaissance_class` / `flooding_class` properties below for the
    # integer ids environment.py actually uses.
    reconnaissance_class_name: str = ""
    flooding_class_name: str = ""

    # Raw preprocessing_metadata.json, kept for logging/provenance only.
    # Its schema differs between datasets (see module docstring) --
    # treat as opaque; never branch application logic on its keys.
    raw_metadata: dict = field(default_factory=dict)

    input_dim: int = field(init=False)
    num_classes: int = field(init=False)

    def __post_init__(self):
        if self.X_train.ndim != 2:
            raise ValueError(
                f"Expected X_train to be 2D [n_samples, n_features], got shape {self.X_train.shape}"
            )
        self.input_dim = self.X_train.shape[1]
        self.num_classes = int(max(self.y_train.max(), self.y_test.max())) + 1

        if self.X_test.shape[1] != self.input_dim:
            raise ValueError(
                f"X_train has {self.input_dim} features but X_test has {self.X_test.shape[1]}"
            )
        if self.X_val is not None and self.X_val.shape[1] != self.input_dim:
            raise ValueError(
                f"X_train has {self.input_dim} features but X_val has {self.X_val.shape[1]}"
            )

        if not self.class_names:
            self.class_names = [f"class_{i}" for i in range(self.num_classes)]
        if len(self.class_names) != self.num_classes:
            raise ValueError(
                f"class_names has {len(self.class_names)} entries ({self.class_names}) but "
                f"{self.num_classes} classes were inferred from y_train/y_test (max label + 1). "
                f"This almost always means the .npy files under this dataset's data_dir are "
                f"stale or from a different preprocessing run -- re-run the notebook, or check "
                f"config.datasets.<name>.data_dir isn't pointing somewhere unexpected."
            )

        if self.reconnaissance_class_name not in self.class_names:
            raise ValueError(
                f"Adapter declared reconnaissance_class_name={self.reconnaissance_class_name!r}, "
                f"which is not among this dataset's actual class_names={self.class_names}. "
                f"Check the adapter's declared semantic roles against the dataset's real "
                f"class_names.npy -- a mismatch here means the notebook's label taxonomy "
                f"changed without the adapter being updated to match."
            )
        if self.flooding_class_name not in self.class_names:
            raise ValueError(
                f"Adapter declared flooding_class_name={self.flooding_class_name!r}, which is "
                f"not among this dataset's actual class_names={self.class_names}. Check the "
                f"adapter's declared semantic roles against the dataset's real class_names.npy."
            )

    @property
    def reconnaissance_class(self) -> int:
        """Integer class id environment.py's HAM defense logic targets."""
        return self.class_names.index(self.reconnaissance_class_name)

    @property
    def flooding_class(self) -> int:
        """Integer class id environment.py's RM defense logic targets."""
        return self.class_names.index(self.flooding_class_name)


class DatasetAdapter(ABC):
    """
    One adapter per raw dataset. `name` is the registry key used by
    config.yaml's `experiment.dataset` and `config.datasets.<name>`.
    """

    name: str = ""

    @abstractmethod
    def load(self, dataset_cfg: dict) -> DatasetBundle:
        """
        dataset_cfg is this dataset's OWN sub-block from config.yaml,
        i.e. config['datasets'][self.name] -- e.g. {'data_dir': ...,
        'max_samples': ..., 'truncation_strategy': ...}.
        """
        raise NotImplementedError
