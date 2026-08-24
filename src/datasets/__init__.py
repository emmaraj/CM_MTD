"""
datasets/__init__.py
---------------------
Adapter registry: routes config.experiment.dataset -> the right
DatasetAdapter. This is the ONLY switch-on-dataset-name in the whole
pipeline -- config_parser.py, environment.py, and models.py never branch
on which dataset is loaded; they only ever see the common DatasetBundle
shape (see base.py). Adding a third dataset later means writing one more
adapter module and adding one line here, not touching any consumer.
"""

from __future__ import annotations

from .base import DatasetAdapter, DatasetBundle
from .cicids2017 import Cicids2017Adapter
from .nidd5g import Nidd5gAdapter

_REGISTRY = {
    Cicids2017Adapter.name: Cicids2017Adapter(),
    Nidd5gAdapter.name: Nidd5gAdapter(),
}


def get_adapter(dataset_name: str) -> DatasetAdapter:
    try:
        return _REGISTRY[dataset_name]
    except KeyError:
        raise ValueError(
            f"Unknown experiment.dataset {dataset_name!r}. Registered datasets: "
            f"{sorted(_REGISTRY.keys())}. To add another dataset: write a new "
            f"DatasetAdapter under src/datasets/, register it in _REGISTRY here, and add "
            f"a matching config.datasets.<name> block in config.yaml."
        )


def registered_dataset_names() -> list:
    return sorted(_REGISTRY.keys())


__all__ = ["DatasetAdapter", "DatasetBundle", "get_adapter", "registered_dataset_names"]
