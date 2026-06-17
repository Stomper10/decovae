"""Back-fill the `cohort` key into already-extracted conditioning sidecars.

``datasets.pooled.derive_conditions`` now emits a ``cohort`` token, but sidecars
written by an earlier ``extract_emb.py`` lack it — so the A-variant (cohort-token
ON) diffusion run would silently never fire its cohort token. This walks an
embeddings/latents directory, reads each ``*.json`` sidecar, and adds
``cond["cohort"]`` derived from the file's subject id (stem ``{cohort}_{eid}_{mod}``,
verified against the manifest cohort vocabulary). Idempotent: sidecars that
already carry ``cohort`` are left untouched. CPU-only — no re-extraction needed.

    python scripts/patch_cohort_into_cond.py --emb_dir /path/to/embeddings [--dry_run]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

# Self-contained (stdlib only) — cohort is the first "_"-token of the sidecar
# stem "{cohort}_{eid}_{mod}", so no repo imports / adapter / cwd dependency.
COHORTS = {"ukb", "ixi", "hcp", "brats", "adni", "oasis"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--emb_dir", required=True,
                   help="embeddings/latents base dir holding *.json cond sidecars.")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sidecars = sorted(glob.glob(os.path.join(args.emb_dir, "**", "*.json"), recursive=True))
    if not sidecars:
        raise SystemExit(f"[patch] no *.json under {args.emb_dir}")

    n_patched = n_skip_present = n_no_cond = n_bad_cohort = 0
    for fp in sidecars:
        with open(fp) as f:
            obj = json.load(f)
        cond = obj.get("cond")
        if not isinstance(cond, dict):
            n_no_cond += 1
            continue
        if cond.get("cohort") is not None:
            n_skip_present += 1
            continue
        # cohort = first "_"-token of the sidecar stem "{cohort}_{eid}_{mod}"
        cohort = os.path.basename(fp)[:-len(".json")].split("_")[0]
        if cohort not in COHORTS:
            n_bad_cohort += 1
            print(f"[patch][WARN] {os.path.basename(fp)}: derived cohort {cohort!r} "
                  f"not in {sorted(COHORTS)} — skipped.")
            continue
        cond["cohort"] = cohort
        obj["cond"] = cond
        if not args.dry_run:
            with open(fp, "w") as f:
                json.dump(obj, f)
        n_patched += 1

    tag = "[dry_run] would patch" if args.dry_run else "patched"
    print(f"[patch] {tag} {n_patched} | already-have-cohort {n_skip_present} | "
          f"no-cond {n_no_cond} | bad-cohort {n_bad_cohort} | total {len(sidecars)}")


if __name__ == "__main__":
    main()
