# Copyright 2025 DecoVAE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Helpers for loading JSON configs."""
from __future__ import annotations

import json


def load_json(path: str) -> dict:
    """Load a JSON config from ``path``.

    Per-user environment overrides (paths, W&B identity, etc.) are sourced from
    ``env.local.sh`` and passed into Python entry points via CLI arguments — see
    each launcher script and the ``--data_dir`` / ``--wandb_entity`` / etc.
    arguments in ``train_*.py`` / ``extract_emb.py`` / ``compute_metric.py``.
    """
    with open(path, "r") as f:
        return json.load(f)
