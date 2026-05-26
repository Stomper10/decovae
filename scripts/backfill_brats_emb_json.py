"""One-shot: backfill missing `_emb.nii.gz.json` metadata sidecars for the
BraTS stage1 embeddings. The sidecars were dropped during extraction because
the pre-fix BraTSAdapter.extract_subject_id over-stripped the session token on
embedding filenames, so every case missed the df id match in _create_json_files.
This regenerates them in place (no re-encoding) using the fixed adapter."""
import os
import glob
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.brats import BraTSAdapter
from extract_emb import _create_json_files, _list_gz_files

adapter = BraTSAdapter()
df = pd.concat([
    adapter.normalize_label_df(pd.read_csv("csv_files/brats_train.csv")),
    adapter.normalize_label_df(pd.read_csv("csv_files/brats_valid.csv")),
], ignore_index=True)
print(f"df_all rows={len(df)}  id_column={adapter.id_column}  sample={df[adapter.id_column].iloc[0]}", flush=True)

root = "/leelabsg/data/wonyoungjang/decodata/brats/stage1"
for exp in ["maisi", "sid1e1", "vad1e-0"]:
    emb_dir = os.path.join(root, exp, "embeddings")
    gz = _list_gz_files(emb_dir)
    before = len(glob.glob(os.path.join(emb_dir, "*.json")))
    _create_json_files(gz, df, adapter)
    after = len(glob.glob(os.path.join(emb_dir, "*.json")))
    print(f"{exp}: gz={len(gz)}  json {before} -> {after}", flush=True)
