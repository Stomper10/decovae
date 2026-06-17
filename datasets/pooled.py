"""Pooled multi-cohort dataset adapter (foundation-model corpus).

Wires the unified pooled manifest + offline preprocessing cache into the
training / extraction scripts. Unlike the per-cohort adapters (which load raw
``.nii.gz`` and preprocess on the fly), this adapter points at the FINISHED
cache produced by ``scripts/preprocess_cache.py``:

    {data_dir}/{cache_key}.npy   — fp16 192^3 in [0,1], single channel
    {data_dir}/{cache_key}.json  — typed conditioning tokens (written at cache time)

so ``data_dir`` here is the cache root (e.g. ``/data/.../decovae_cache``) and the
label CSV is ``csv_files/pooled_manifest_{split}.csv``. Because the volumes are
already preprocessed, the VAE transform must run in ``cached=True`` mode
(``configs/pooled/dataset.json: "cached_input": true``).

Conditioning is a TYPED TOKEN SET (modality / age / sex / dx / severity), not a
fixed float vector — ``derive_conditions`` returns a dict whose absent entries
are ``None`` (token not emitted). The VAE itself is unconditional (it consumes
only the ``image``); the token set is consumed downstream by the diffusion UNet.

``vae_only`` rows (BraTS-T1c) train the shared VAE but are NOT diffusion
generation targets — see ``include_vae_only``.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .base import DatasetAdapter

# Manifest columns emitted as conditioning tokens (severity == cdrsb). `cohort`
# is also emitted (always present) so the A-variant (cohort-token ON) ablation
# can condition on it; the B-variant model config simply omits it from its
# attribute list, so the same cond sidecar serves both. `site` stays bookkeeping.
_TOKEN_CONT = ("age", "cdrsb")           # continuous scalars
_TOKEN_CAT = ("modality", "sex", "dx")   # categorical strings (cohort added separately)


def _num_or_none(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _str_or_none(v):
    s = "" if v is None else str(v).strip()
    return s if s and s.lower() != "nan" else None


class PooledAdapter(DatasetAdapter):
    """Adapter over the pooled preprocessing cache + manifest.

    Parameters
    ----------
    include_vae_only:
        When True (default — VAE stage) every manifest row is used, including
        ``vae_only`` rows (BraTS-T1c) so the shared VAE encoder/decoder sees
        T1c. When False (diffusion stage) ``vae_only`` rows are dropped so the
        diffusion modality vocabulary stays {T1, T2, FLAIR}.
    """

    name = "pooled"
    modality = "mri"
    # per-volume id = cache_key basename "{cohort}_{eid}_{mod}" (the cache_key
    # itself has a "{cohort}/" dir prefix, so we join on a derived `sid` column
    # added by normalize_label_df).
    id_column = "sid"

    def __init__(self, include_vae_only: bool = True):
        self.include_vae_only = include_vae_only

    # -- manifest / io --------------------------------------------------------
    def _read(self, csv_path: str, n: int | None) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        if not self.include_vae_only and "vae_only" in df.columns:
            df = df[df["vae_only"] == 0].reset_index(drop=True)
        if n is not None:
            df = df[:n]
        return df

    # suffixes added downstream of the cache .npy (embedding / latent files).
    _SUFFIXES = ("_emb.nii.gz", "_mu.npy", "_sigma.npy", ".npy", ".nii.gz")

    def extract_subject_id(self, image_path: str) -> str:
        # Works on the cache .npy AND on the derived embedding/latent files, all
        # of which share the basename stem "{cohort}_{eid}_{mod}".
        b = os.path.basename(image_path)
        for suf in self._SUFFIXES:
            if b.endswith(suf):
                return b[: -len(suf)]
        return b

    def load_manifest(self, csv_path: str, data_dir: str,
                      n: int | None = None) -> list[dict]:
        df = self._read(csv_path, n)
        return [
            {"image": os.path.join(data_dir, f"{key}.npy"), "class": self.modality}
            for key in df["cache_key"]
        ]

    def load_manifest_stratified(self, csv_path: str, data_dir: str,
                                 n: int, stage: str = "vae",
                                 seed: int = 0) -> list[dict]:
        """Balanced cell-stratified manifest of (about) ``n`` rows.

        The pooled valid CSV is cohort-ordered and UKB-dominated (≈76%), so a
        plain first-``n`` slice (``load_manifest(n=...)``) yields an all-UKB
        in-loop validation set. This instead round-robins across sampling cells
        (``cell_of(row, stage)``) after a deterministic per-cell shuffle, so
        every cell — including small cohorts (ixi / oasis) — is represented
        roughly equally (``n // num_cells`` each, remainder to the first cells).
        Used only for the recon-monitoring val loop; final FID / checkpoint
        selection runs over the full valid split in ``compute_metric``.
        """
        df = self._read(csv_path, None)
        cells = {}
        for pos, (_, row) in enumerate(df.iterrows()):
            cells.setdefault(self.cell_of(row, stage), []).append(pos)
        rng = np.random.default_rng(seed)
        for c in cells:
            rng.shuffle(cells[c])
        order = sorted(cells)  # deterministic cell order
        picked: list[int] = []
        i = 0
        while len(picked) < n and any(cells[c] for c in order):
            c = order[i % len(order)]
            if cells[c]:
                picked.append(cells[c].pop())
            i += 1
        sub = df.iloc[sorted(picked)]
        # "cell" rides along for per-cohort val logging; the VAE transform's
        # SelectItemsd(keys=["image"]) drops it before tensors are built.
        return [
            {"image": os.path.join(data_dir, f"{r.cache_key}.npy"),
             "class": self.modality, "cell": self.cell_of(r, stage)}
            for _, r in sub.iterrows()
        ]

    # -- imbalance sampling (§4.3) -------------------------------------------
    @staticmethod
    def cell_of(row, stage: str) -> str:
        """Sampling cell. VAE: cohort×modality. Diffusion: cohort×modality×dx."""
        base = f"{row['cohort']}|{row['modality']}"
        if stage == "diffusion":
            dx = _str_or_none(row.get("dx")) or "na"
            return f"{base}|{dx}"
        return base

    def cell_labels(self, csv_path: str, stage: str, n: int | None = None) -> list[str]:
        """Cell label per row, aligned with load_manifest order (same filter)."""
        df = self._read(csv_path, n)
        return [self.cell_of(r, stage) for _, r in df.iterrows()]

    def sid_to_cell(self, csv_path: str, stage: str) -> dict:
        """{sid: cell} lookup (full manifest) — for embedding-file sample sets."""
        df = self.normalize_label_df(self._read(csv_path, None))
        return {r["sid"]: self.cell_of(r, stage) for _, r in df.iterrows()}

    # -- conditioning ---------------------------------------------------------
    def normalize_label_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # Add the per-volume join key `sid` = cache_key basename (drops the
        # "{cohort}/" dir prefix) so embeddings/latents join back to the row.
        # Typed-token continuous normalisation (age/severity scaling) is the
        # token encoder's job (needs pooled-train stats) → no norm_* columns here.
        df = df.copy()
        df["sid"] = df["cache_key"].astype(str).str.rsplit("/", n=1).str[-1]
        return df

    def derive_conditions(self, row: pd.Series) -> dict:
        """Typed token set for one volume. Absent value → None (token not emitted).

        Returns a dict (not the legacy list[float]); the diffusion token-set
        encoder consumes present tokens only. Continuous values are raw here
        (age in years, cdrsb score) — normalised downstream.
        """
        tokens: dict = {}
        for k in _TOKEN_CAT:
            tokens[k] = _str_or_none(row.get(k))
        # cohort: always present; used as a token only by the A-variant model
        # config (B omits it from its attribute list and ignores this key).
        tokens["cohort"] = _str_or_none(row.get("cohort"))
        for k in _TOKEN_CONT:
            tokens[k] = _num_or_none(row.get(k))
        return tokens
