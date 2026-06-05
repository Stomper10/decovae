"""Dataset adapter registry.

Adapters encapsulate dataset-specific code (subject-id extraction, manifest
construction, label/cond derivation) so that the training and analysis scripts
can stay dataset-agnostic.

Usage:
    from datasets import get_adapter
    adapter = get_adapter("ukb_20252")
    files = adapter.load_manifest(csv_path, data_dir)
"""
from __future__ import annotations

from .base import DatasetAdapter
from .brats import BraTSAdapter
from .ixi import IXIAdapter
from .pooled import PooledAdapter
from .ukb_20252 import UKB20252Adapter

_REGISTRY: dict[str, type[DatasetAdapter]] = {
    "brats": BraTSAdapter,
    "ixi": IXIAdapter,
    "pooled": PooledAdapter,
    "ukb_20252": UKB20252Adapter,
}


def get_adapter(name: str, **kwargs) -> DatasetAdapter:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown dataset adapter '{name}'. "
            f"Registered: {sorted(_REGISTRY)}. "
            f"Add a new entry to datasets/__init__.py to register one."
        )
    return _REGISTRY[name](**kwargs)


__all__ = ["DatasetAdapter", "get_adapter"]
