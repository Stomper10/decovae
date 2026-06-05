"""Typed metadata token-set encoder for diffusion conditioning.

Each present attribute k contributes a token

    token_k = key_emb[k] + value_enc_k(v_k)   ∈ R^cond_dim

(FT-Transformer-style feature tokenization). Categorical attributes use a
per-attribute embedding table; continuous attributes a small MLP over the
[0,1]-normalised scalar. Present tokens (presence mask) are mean-pooled into a
single vector; an EMPTY set — classifier-free-guidance full-drop, or a volume
with no metadata — maps to a learned null embedding. The pooled vector is
projected to `output_dim` so it can be concatenated into the UNet
time-embedding exactly like the legacy `meta_layer`.

Attribute ordering convention (must match the dataloader's presence/cat/cont
layout): all categorical attributes first (schema order), then all continuous
attributes (schema order). Index 0 is therefore the first categorical attribute
(modality), which the CFG per-token drop keeps.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def split_attributes(attributes: list[dict]):
    """(cat_attrs, cont_attrs) in the canonical cats-then-conts order."""
    cat = [a for a in attributes if a["type"] == "cat"]
    cont = [a for a in attributes if a["type"] == "cont"]
    return cat, cont


def encode_token_set(cond: dict, attributes: list[dict]):
    """Typed-token dict → (cat_idx, cont_val, presence) python lists, in the
    canonical cats-then-conts order. Absent value → presence False (and a
    placeholder 0, masked out by the encoder — never a fabricated value).
    Shared by the train_UNET dataloader and compute_metric generation so the
    conditioning representation is identical at train and inference time."""
    cat, cont = split_attributes(attributes)
    cat_idx, presence = [], []
    for a in cat:
        vmap = {v: i for i, v in enumerate(a["vocab"])}
        v = cond.get(a["name"])
        ok = v is not None and v in vmap
        cat_idx.append(vmap.get(v, 0) if ok else 0)
        presence.append(ok)
    cont_val = []
    for a in cont:
        v = cond.get(a["name"])
        ok = v is not None
        try:
            x = (float(v) - a["min"]) / (a["max"] - a["min"]) if ok else 0.0
        except (TypeError, ValueError):
            x, ok = 0.0, False
        cont_val.append(min(max(x, 0.0), 1.0))
        presence.append(ok)
    return cat_idx, cont_val, presence


class TokenSetEncoder(nn.Module):
    def __init__(self, attributes: list[dict], cond_dim: int, output_dim: int,
                 pool: str = "mean"):
        super().__init__()
        self.cat, self.cont = split_attributes(attributes)
        self.n_cat = len(self.cat)
        self.n_cont = len(self.cont)
        self.n_attr = self.n_cat + self.n_cont
        self.cond_dim = cond_dim
        if pool != "mean":
            raise NotImplementedError(f"pool={pool!r} not supported (only 'mean').")
        self.pool = pool

        self.key_emb = nn.Embedding(self.n_attr, cond_dim)
        self.cat_emb = nn.ModuleList(
            [nn.Embedding(len(a["vocab"]), cond_dim) for a in self.cat]
        )
        self.cont_mlp = nn.ModuleList(
            [nn.Sequential(nn.Linear(1, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
             for _ in self.cont]
        )
        self.null_emb = nn.Parameter(torch.zeros(cond_dim))
        self.out_proj = nn.Linear(cond_dim, output_dim)

    def forward(self, cat_idx: torch.Tensor, cont_val: torch.Tensor,
                presence: torch.Tensor) -> torch.Tensor:
        """cat_idx (B,n_cat) long, cont_val (B,n_cont) float, presence (B,n_attr) bool/float."""
        B = presence.shape[0]
        tokens = []
        for j in range(self.n_cat):
            key = self.key_emb.weight[j].unsqueeze(0)          # (1,cond_dim)
            val = self.cat_emb[j](cat_idx[:, j])               # (B,cond_dim)
            tokens.append(key + val)
        for j in range(self.n_cont):
            key = self.key_emb.weight[self.n_cat + j].unsqueeze(0)
            val = self.cont_mlp[j](cont_val[:, j:j + 1])       # (B,cond_dim)
            tokens.append(key + val)
        tokens = torch.stack(tokens, dim=1)                    # (B,n_attr,cond_dim)

        m = presence.to(tokens.dtype).unsqueeze(-1)            # (B,n_attr,1)
        cnt = m.sum(dim=1)                                     # (B,1)
        pooled = (tokens * m).sum(dim=1) / cnt.clamp(min=1.0)  # masked mean
        # empty set (no present token) → learned null embedding
        empty = (cnt.squeeze(-1) == 0).unsqueeze(-1)           # (B,1)
        null = self.null_emb.to(tokens.dtype).unsqueeze(0).expand(B, -1)
        pooled = torch.where(empty, null, pooled)
        return self.out_proj(pooled)                           # (B,output_dim)
