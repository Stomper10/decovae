# Copyright 2025 DecoVAE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Helpers for loading JSON configs with per-user local overrides."""
from __future__ import annotations

import copy
import json
import os


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_json_with_local(path: str) -> dict:
    """Load a JSON config and apply a sibling ``*.local.json`` override if present.

    Given ``foo/bar.json``, also looks for ``foo/bar.local.json`` and deep-merges
    it on top of the base config. ``*.local.json`` files are gitignored, so each
    user can keep their environment-specific paths / credentials private without
    editing the public configs that ship with the repository.
    """
    with open(path, "r") as f:
        config = json.load(f)
    base, ext = os.path.splitext(path)
    local_path = f"{base}.local{ext}"
    if os.path.isfile(local_path):
        with open(local_path, "r") as f:
            local = json.load(f)
        config = deep_merge(config, local)
    return config
