"""Build train/val CSVs for the adherence predictors from the pooled manifest.

Emits a ``rel_path`` + target-column CSV that ``downstream.attr_dataset`` /
``brain_age_dataset`` can ingest, with per-target filtering:

  dx  : cohorts with clinical labels only (default adni,oasis) and
        dx in {healthy(=CN), MCI, AD} — tumor / NA dropped. label_map for the
        trainer: '{"healthy":0,"MCI":1,"AD":2}'.
  sex : every volume with a non-missing sex (drops brats, a few ixi/adni).
  age : every volume with a non-missing age (pooled brain-age regressor).
  cohort : ukb / ixi / hcp / brats / adni / oasis — the A-config conditioning
        vocabulary (model_fm_cohort.json), restricted to T1/T2/FLAIR for the same
        reason modality is: the generator can never emit T1c, so scoring it would
        measure a class that was never generated. label_map for the trainer:
        '{"ukb":0,"ixi":1,"hcp":2,"brats":3,"adni":4,"oasis":5}'. Wildly unbalanced
        (ukb 39,807 vs ixi 928, a 43x spread) — pass CLASS_WEIGHTED=1.
        READ THE RESULT PER MODALITY. cohort and modality are confounded in this
        corpus: T2 exists only in brats/hcp/ixi/oasis and FLAIR only in
        adni/brats/ukb, so a classifier can score well by reading modality and
        excluding cohorts it rules out. T1 is the ONLY slice carrying all six
        cohorts, and is therefore the clean read for cohort adherence.
  modality : T1 / T2 / FLAIR only. T1c is EXCLUDED on purpose — brats_T1c is
        vae_only=1 and sits outside the diffusion conditioning vocabulary
        (model_fm.json modality vocab is exactly ["T1","T2","FLAIR"]), so a 4-way
        predictor would be scoring a class the generator was never able to emit.
        label_map for the trainer: '{"T1":0,"T2":1,"FLAIR":2}'. Classes are very
        unbalanced in train (T1 26,146 / FLAIR 21,987 / T2 3,036) — pass
        CLASS_WEIGHTED=1. This is the PRIMARY adherence axis: modality is present on
        100% of volumes and is the one attribute CFG never drops (cfg_drop_presence
        keep_idx=(0,)), so it is the only condition the model always received.

``--path_col`` picks the volume the predictor trains on:
  cache_key (default) → preprocessed ``.npy`` (SAME space the VAE/diffusion sees,
            so the predictor matches generation space) → rel_path = "<cache_key>.npy".
  src_path            → raw source NIfTI → rel_path = "<src_path>" (set data_dir="").

NOTE: train the predictor in the space it will be EVALUATED in (generated volumes
are VAE-decoded ≈ preprocessed space). cache_key is the safer default; see the
adherence-eval normalization note before trusting numbers across .npy↔.nii.gz.
"""
from __future__ import annotations

import argparse

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="pooled_manifest_{train,valid}.csv")
    p.add_argument("--target", required=True,
                   choices=["dx", "sex", "age", "modality", "cohort"])
    p.add_argument("--out_csv", required=True)
    p.add_argument("--path_col", default="cache_key", choices=["cache_key", "src_path"])
    p.add_argument("--cohorts", default=None,
                   help="comma list to restrict cohorts (default: dx→adni,oasis; else all).")
    p.add_argument("--dx_labels", default="healthy,MCI,AD",
                   help="dx only: kept classes (drops others incl tumor/NA).")
    p.add_argument("--modality_labels", default="T1,T2,FLAIR",
                   help="modality only: kept classes (T1c is excluded by default).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest)

    cohorts = args.cohorts.split(",") if args.cohorts else (
        ["adni", "oasis"] if args.target == "dx" else None)
    if cohorts:
        df = df[df["cohort"].isin(cohorts)]

    if args.target == "dx":
        keep = args.dx_labels.split(",")
        df = df[df["dx"].isin(keep)]
    elif args.target == "sex":
        df = df[df["sex"].notna()]
    elif args.target == "modality":
        df = df[df["modality"].isin(args.modality_labels.split(","))]
    elif args.target == "cohort":
        # Same modality restriction as the modality judge, and for the same
        # reason: these are the classes the diffusion model can actually emit.
        df = df[df["modality"].isin(args.modality_labels.split(","))]
    else:  # age
        df = df[df["age"].notna()]

    if args.path_col == "cache_key":
        rel = df["cache_key"].astype(str) + ".npy"
    else:
        rel = df["src_path"].astype(str)

    # cohort/modality ride along for slicing the results later. When the target IS
    # modality the dict would collapse two identical keys into one column, so build
    # it explicitly instead of relying on literal ordering.
    cols = {"rel_path": rel.values, args.target: df[args.target].values}
    for extra in ("cohort", "modality"):
        cols.setdefault(extra, df[extra].values)
    out = pd.DataFrame(cols)
    out.to_csv(args.out_csv, index=False)

    print(f"[build_adherence_csv] target={args.target} path_col={args.path_col} "
          f"cohorts={cohorts or 'all'} -> {args.out_csv} ({len(out)} rows)")
    if args.target in ("dx", "sex", "modality", "cohort"):
        print("  class distribution:")
        print(out[args.target].value_counts().to_string())
    if args.target == "cohort":
        print("  cohort x modality (read the eval per modality — they are confounded):")
        print(pd.crosstab(out["cohort"], out["modality"]).to_string())
    # Guard on the column, not on the target. `age` is emitted only when it IS the
    # target -- the cols dict carries rel_path, the target, cohort and modality and
    # nothing else -- so an `else` here crashed every dx/sex/modality build. It was
    # introduced with the cohort branch and hid because the CSV is written before this
    # line, so the file looked fine while the script exited non-zero.
    if "age" in out.columns:
        print(f"  age range: {out['age'].min():.1f}–{out['age'].max():.1f}  "
              f"mean {out['age'].mean():.1f}  n={len(out)}")


if __name__ == "__main__":
    main()
