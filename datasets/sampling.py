"""Temperature-weighted distributed sampling for the pooled corpus.

The pooled corpus is heavily imbalanced (UKB ~76%); naive uniform sampling lets
the dominant cohort/modality dominate. We oversample rare cells with a
temperature τ: per-sample weight ``w_i ∝ n_c^(τ-1)`` where ``n_c`` is the size of
sample i's cell. τ=1 → uniform (no reweighting); τ=0 → fully balanced across
cells; τ=0.5 → intermediate. Cells are stage-specific (§4.3):
  VAE (unconditional)  : cohort × modality
  Diffusion (conditioned): cohort × modality × dx
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Sampler


def temperature_weights(cells, tau: float = 0.5) -> np.ndarray:
    """Per-sample weights w_i ∝ n_c^(tau-1) (unnormalised; multinomial normalises)."""
    n = Counter(cells)
    return np.asarray([n[c] ** (tau - 1.0) for c in cells], dtype=np.float64)


class DistributedWeightedSampler(Sampler):
    """Weighted (with-replacement) sampling, distributed + per-epoch reshuffle.

    Mirrors ``DistributedSampler`` sharding: one global weighted draw of
    ``total_size`` indices (same RNG on every rank, seeded by epoch) is strided
    by rank so ranks see disjoint slices. ``set_epoch`` reshuffles. Drop-in for
    the ``infinite_loader(loader, sampler, epoch)`` pattern used by the trainers.
    """

    def __init__(self, weights, num_replicas: int, rank: int, seed: int = 0):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        n = len(self.weights)
        self.num_samples = int(math.ceil(n / num_replicas))
        self.total_size = self.num_samples * num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        idx = torch.multinomial(self.weights, self.total_size, replacement=True, generator=g)
        idx = idx[self.rank:self.total_size:self.num_replicas]
        return iter(idx.tolist())
