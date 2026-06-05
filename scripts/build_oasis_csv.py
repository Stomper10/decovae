#!/usr/bin/env python3
"""Build OASIS-3 cohort CSVs (pooled generative corpus + external test).
- T1: all subjects (100% isotropic).
- T2: isotropic subset only (z<=1.5mm); thick-slice 2D dropped (see plan §3.1/§6).
- FLAIR: NOT built (excluded from generative training; thick-slice, qualitative-only).
Subject-level dx-stratified 80/10/10 split (1 session/subject = baseline).
T2 follows the SAME subject split as T1 (no cross-modality leakage).
Schema mirrors ADNI: eid,rel_path,site,modality,age,sex,group,dx,cdrsb,mmse.
"""
import csv, os, glob, random, re
import nibabel as nib

ROOT = "/data/wonyoungjang/OASIS-3"
DATA = f"{ROOT}/data"
OUT  = "/data/wonyoungjang/decovae/csv_files"
COND = f"{ROOT}/oasis3_baseline_conditions.csv"
ISO_THRESH = 1.5  # mm; max voxel dim <= this == isotropic
SEED = 42

def max_zoom(path):
    try:
        return max(nib.load(path).header.get_zooms()[:3])
    except Exception:
        return 999.0

MIN_SLICES = 32  # reject 2D scouts / localizers (e.g. shape (484,484,1))

def is_3d_volume(path):
    """True if every spatial dim has >= MIN_SLICES (drops scout/localizer slices)."""
    try:
        return min(nib.load(path).shape[:3]) >= MIN_SLICES
    except Exception:
        return False

def pick(cands, want_iso):
    """Pick best file: prefer non-run-02, then most isotropic, then name-sorted.
    If want_iso, only keep iso candidates (return None if none).
    Always rejects 2D scouts/localizers (min spatial dim < MIN_SLICES)."""
    if not cands:
        return None
    cands = [p for p in cands if is_3d_volume(p)]
    if not cands:
        return None
    scored = []
    for p in sorted(cands):
        z = max_zoom(p)
        run2 = 1 if "run-02" in p else 0
        scored.append((run2, round(z, 3), p, z))
    if want_iso:
        scored = [s for s in scored if s[3] <= ISO_THRESH]
        if not scored:
            return None
    scored.sort(key=lambda s: (s[0], s[1], s[2]))  # non-run2, most-iso, name
    return scored[0][2]

# ---- build a file index ONCE (single walk), keyed by (subject, day, modality) ----
# filename e.g. sub-OAS30001_ses-d0129_run-01_T1w.nii.gz  (also 'sess-' variant)
FNRE = re.compile(r"sub-(OAS\d+)_ses+-(?:d)?(\d+).*?_(T1w|T2w|FLAIR)\.nii\.gz$")
INDEX = {}  # (subject, 'd'+day, mod) -> [paths]
for dp, _, files in os.walk(DATA):
    for fn in files:
        if not fn.endswith(".nii.gz"):
            continue
        m = FNRE.search(fn)
        if not m:
            continue
        subj, day, mod = m.group(1), "d"+m.group(2), m.group(3)
        INDEX.setdefault((subj, day, mod), []).append(os.path.join(dp, fn))

def find_files(mr_id, mod):
    # mr_id = OAS30001_MR_d0129 -> subject OAS30001, day d0129
    subject, day = mr_id.split("_MR_")
    return INDEX.get((subject, day, mod), [])

# 1) load clinical
rows = {}
with open(COND) as f:
    for r in csv.DictReader(f):
        rows[r["mr_id"]] = r

# 2) resolve files per session
SEXMAP = {"1": "M", "2": "F"}
recs = {}  # mr_id -> {t1, t2(iso or None), meta}
for mr_id, r in rows.items():
    t1 = pick(find_files(mr_id, "T1w"), want_iso=False)
    t2 = pick(find_files(mr_id, "T2w"), want_iso=True)
    recs[mr_id] = {"t1": t1, "t2": t2, "r": r}

n_t1 = sum(1 for v in recs.values() if v["t1"])
n_t2 = sum(1 for v in recs.values() if v["t2"])
print(f"sessions={len(recs)}  with_T1={n_t1}  with_isoT2={n_t2}")

# 3) subject-level dx-stratified split (1 session/subject)
subjects = sorted(recs.keys())  # mr_id == one subject's baseline
by_dx = {}
for mr_id in subjects:
    dx = recs[mr_id]["r"]["dx"] or "unknown"
    by_dx.setdefault(dx, []).append(mr_id)

rng = random.Random(SEED)
split = {}  # mr_id -> train/valid/test
for dx, ids in by_dx.items():
    ids = ids[:]
    rng.shuffle(ids)
    n = len(ids); n_test = round(n*0.10); n_val = round(n*0.10)
    for i, mid in enumerate(ids):
        split[mid] = "test" if i < n_test else ("valid" if i < n_test+n_val else "train")
print("split by dx:", {dx: len(v) for dx, v in by_dx.items()})

# 4) write CSVs
HEADER = ["eid","rel_path","site","modality","age","sex","group","dx","cdrsb","mmse"]
def row_for(mr_id, path, modality):
    r = recs[mr_id]["r"]
    rel = os.path.relpath(path, DATA)
    return [r["subject"], rel, "OASIS", modality, r["age"],
            SEXMAP.get(r["sex"], "unknown"), r["dx"] or "unknown", r["dx"] or "unknown",
            r["cdrsb"], r["mmse"]]

for modality, key, want in [("T1","t1",True), ("T2","t2",True)]:
    buckets = {"train": [], "valid": [], "test": []}
    for mr_id in subjects:
        p = recs[mr_id][key]
        if not p:
            continue
        buckets[split[mr_id]].append(row_for(mr_id, p, modality))
    for sp, data in buckets.items():
        data.sort(key=lambda x: x[0])
        fn = f"{OUT}/oasis_{modality}_{sp}.csv"
        with open(fn, "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(HEADER); w.writerows(data)
        print(f"  wrote {fn}: {len(data)}")

print("=== OASIS_CSV_DONE ===")
