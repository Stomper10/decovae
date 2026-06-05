#!/usr/bin/env python3
"""Build the pooled foundation-model manifest from per-cohort×modality CSVs.

One row per volume (subject × modality). Harmonizes heterogeneous per-cohort
schemas into a single token-set conditioning manifest:

    cohort, subject_id, modality, split, src_path, is_stripped, is_registered,
    site, age, age_binned, sex, dx, cdrsb, mmse, vae_only, cache_key

Token-emission rule = column presence: an EMPTY cell means that token is not
emitted for that volume (modality/dx always present; age/sex/severity may be
absent). dx vocabulary = {healthy, MCI, AD, tumor} (ADNI/OASIS "CN" already
merged to healthy at their build time; OASIS "unknown" dx → absent token).

Pooled corpus = 14 cohort×modality cells:
    UKB{T1,FLAIR} IXI{T1,T2} HCP{T1,T2} BraTS{T1,T1c,T2,FLAIR} ADNI{T1,FLAIR} OASIS{T1,T2}
  `vae_only` flag: BraTS-T1c rows are VAE-only (vae_only=1) — included so the
  shared pooled VAE encoder/decoder sees T1c (reused by the §6 tumor-seg
  ControlNet pipeline which generates 4-modal {T1,T1c,T2,FLAIR}), but EXCLUDED
  from the diffusion modality vocabulary {T1,T2,FLAIR} (T1c⟂tumor confound:
  T1c exists only in BraTS = all dx=tumor). Diffusion consumers filter
  vae_only==0; VAE consumers use all rows. (OASIS-FLAIR stays out entirely.)

Separately writes brats_tumor_manifest_{split}.csv: per-subject 4-modal
{T1,T1c,T2,FLAIR} + seg for the BraTS-specific tumor-seg pipeline (§6).
"""
import os
import numpy as np
import pandas as pd

CSV = "/data/wonyoungjang/decovae/csv_files"
SPLITS = ["train", "valid", "test"]

# cohort×modality -> source data root (rel_path is relative to this)
ROOT = {
    ("ukb", "T1"):    "/data/wonyoungjang/20252_unzip",
    ("ukb", "FLAIR"): "/data/wonyoungjang/20253_unzip",
    ("ixi", "T1"):    "/data/wonyoungjang/IXI",
    ("ixi", "T2"):    "/data/wonyoungjang/IXI",
    ("hcp", "T1"):    "/data/wonyoungjang/HCP",
    ("hcp", "T2"):    "/data/wonyoungjang/HCP",
    ("brats", "T1"):    "/data/wonyoungjang/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
    ("brats", "T1c"):   "/data/wonyoungjang/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
    ("brats", "T2"):    "/data/wonyoungjang/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
    ("brats", "FLAIR"): "/data/wonyoungjang/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
    ("adni", "T1"):    "/data/wonyoungjang/ADNI",
    ("adni", "FLAIR"): "/data/wonyoungjang/ADNI",
    ("oasis", "T1"): "/data/wonyoungjang/OASIS-3/data",
    ("oasis", "T2"): "/data/wonyoungjang/OASIS-3/data",
}

# pooled corpus (excludes oasis FLAIR). BraTS-T1c is included but VAE-only.
POOLED = [
    ("ukb", "T1"), ("ukb", "FLAIR"),
    ("ixi", "T1"), ("ixi", "T2"),
    ("hcp", "T1"), ("hcp", "T2"),
    ("brats", "T1"), ("brats", "T1c"), ("brats", "T2"), ("brats", "FLAIR"),
    ("adni", "T1"), ("adni", "FLAIR"),
    ("oasis", "T1"), ("oasis", "T2"),
]

# cells included for the VAE only (not in the diffusion modality vocabulary)
VAE_ONLY = {("brats", "T1c")}

DX_BY_COHORT = {"ukb": "healthy", "ixi": "healthy", "hcp": "healthy",
                "brats": "tumor"}  # adni/oasis read from CSV column
HCP_AGE_MID = {"22-25": 23.5, "26-30": 28.0, "31-35": 33.0, "36+": 37.0}


def num(x):
    """Coerce to float; '' / 'unknown' / NaN -> np.nan."""
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def harmonize_sex(cohort, raw):
    if cohort in ("ukb", "ixi"):           # 0=F, 1=M (verified)
        v = num(raw)
        return "" if np.isnan(v) else ("M" if v == 1 else "F")
    s = str(raw).strip().upper()
    return s if s in ("M", "F") else ""     # ADNI 'X' / BraTS 'unknown' -> absent


def harmonize_age(cohort, raw):
    if cohort == "hcp":
        return HCP_AGE_MID.get(str(raw).strip(), np.nan)
    return num(raw)


def harmonize_dx(cohort, row):
    if cohort in DX_BY_COHORT:
        return DX_BY_COHORT[cohort]
    dx = str(row.get("dx", "")).strip()     # adni/oasis
    return dx if dx in ("healthy", "MCI", "AD") else ""   # 'unknown' -> absent


def subject_id(cohort, eid):
    return f"{cohort}_{eid}"


def build_pooled():
    rows = {s: [] for s in SPLITS}
    for cohort, mod in POOLED:
        root = ROOT[(cohort, mod)]
        binned = 1 if cohort == "hcp" else 0
        stripped = 1 if cohort == "brats" else 0
        registered = 1 if cohort == "brats" else 0
        vae_only = 1 if (cohort, mod) in VAE_ONLY else 0
        for split in SPLITS:
            df = pd.read_csv(f"{CSV}/{cohort}_{mod}_{split}.csv")
            for _, r in df.iterrows():
                sid = subject_id(cohort, r["eid"])
                site = str(r["site"]) if "site" in df and pd.notna(r.get("site")) else cohort
                age = harmonize_age(cohort, r.get("age"))
                rows[split].append({
                    "cohort": cohort,
                    "subject_id": sid,
                    "modality": mod,
                    "split": split,
                    "src_path": os.path.join(root, r["rel_path"]),
                    "is_stripped": stripped,
                    "is_registered": registered,
                    "site": site,
                    "age": age,
                    "age_binned": binned,
                    "sex": harmonize_sex(cohort, r.get("sex")),
                    "dx": harmonize_dx(cohort, r),
                    "cdrsb": num(r.get("cdrsb")) if "cdrsb" in df else np.nan,
                    "mmse": num(r.get("mmse")) if "mmse" in df else np.nan,
                    "vae_only": vae_only,
                    "cache_key": f"{cohort}/{sid}_{mod}",
                })
    cols = ["cohort", "subject_id", "modality", "split", "src_path",
            "is_stripped", "is_registered", "site", "age", "age_binned",
            "sex", "dx", "cdrsb", "mmse", "vae_only", "cache_key"]
    for split in SPLITS:
        out = pd.DataFrame(rows[split], columns=cols)
        path = f"{CSV}/pooled_manifest_{split}.csv"
        out.to_csv(path, index=False, lineterminator="\n")
        print(f"  wrote {path}: {len(out)}")
    return cols


def build_brats_tumor():
    """Per-subject 4-modal {T1,T1c,T2,FLAIR} + seg for tumor-seg pipeline."""
    root = ROOT[("brats", "T1")]
    mods = ["T1", "T1c", "T2", "FLAIR"]
    for split in SPLITS:
        per_mod = {m: pd.read_csv(f"{CSV}/brats_{m}_{split}.csv").set_index("eid")
                   for m in mods}
        base = per_mod["T1"]
        recs = []
        for eid, r in base.iterrows():
            sid = subject_id("brats", eid)
            rec = {"subject_id": sid, "split": split, "dx": "tumor",
                   "seg_path": os.path.join(root, r["rel_path_seg"])}
            ok = True
            for m in mods:
                if eid not in per_mod[m].index:
                    ok = False
                    break
                rec[f"{m.lower()}_path"] = os.path.join(root, per_mod[m].loc[eid, "rel_path"])
            if ok:
                recs.append(rec)
        cols = ["subject_id", "split", "t1_path", "t1c_path", "t2_path",
                "flair_path", "seg_path", "dx"]
        out = pd.DataFrame(recs, columns=cols)
        path = f"{CSV}/brats_tumor_manifest_{split}.csv"
        out.to_csv(path, index=False, lineterminator="\n")
        print(f"  wrote {path}: {len(out)}")


if __name__ == "__main__":
    print("=== pooled manifest ===")
    build_pooled()
    print("=== brats tumor manifest ===")
    build_brats_tumor()
    print("=== POOLED_MANIFEST_DONE ===")
