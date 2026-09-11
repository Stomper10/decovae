#!/usr/bin/env python3
"""Aggregate §3b per-source CSVs into a single summary CSV.

Reads:  journal_plan/tumor_mode/<label>.csv  (per-volume rows produced by
        scripts/eval_tumor_mode_avg.py, 10 sources: 1 real + 9 gen arms).

Writes: journal_plan/results_tumor_mode_summary.csv  — one row per source
        with detection rate, median/IQR for connected-component count,
        dispersion, and volume (WT/TC/ET).

Also refreshes journal_plan/results_seg_domain_shift_summary.csv from the
existing per-case §3a CSV (mean native vs pooled dice, per region).
"""
from __future__ import annotations
import csv, math, statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TM_DIR = REPO / "journal_plan" / "tumor_mode"
TM_OUT = REPO / "journal_plan" / "results_tumor_mode_summary.csv"
SEG_IN = REPO / "journal_plan" / "results_seg_domain_shift.csv"
SEG_OUT = REPO / "journal_plan" / "results_seg_domain_shift_summary.csv"

REGIONS = ("WT", "TC", "ET")


def _floats(rows, key):
    out = []
    for r in rows:
        v = r.get(key, "")
        if v in ("", "nan"):
            continue
        try:
            f = float(v)
            if not math.isnan(f):
                out.append(f)
        except ValueError:
            pass
    return out


def _ints(rows, key):
    return [int(x) for x in _floats(rows, key)]


def _median_iqr(xs):
    if not xs:
        return float("nan"), float("nan"), float("nan")
    xs = sorted(xs)
    n = len(xs)
    med = xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2
    q1 = xs[n // 4]
    q3 = xs[(3 * n) // 4]
    return med, q1, q3


def _mean(xs):
    return st.mean(xs) if xs else float("nan")


def harvest_tumor_mode() -> int:
    fields = ["source", "n", "any_detected_pct"]
    for reg in REGIONS:
        fields += [
            f"det_pct_{reg}",
            f"vol_med_{reg}", f"vol_q1_{reg}", f"vol_q3_{reg}",
            f"cc_med_{reg}",  f"cc_q1_{reg}",  f"cc_q3_{reg}",
            f"disp_med_{reg}", f"disp_q1_{reg}", f"disp_q3_{reg}",
            f"largest_cc_frac_med_{reg}",
        ]

    rows_out = []
    for src in sorted(TM_DIR.glob("*.csv")):
        with src.open() as f:
            per = list(csv.DictReader(f))
        n = len(per)
        row = {"source": src.stem, "n": n}
        det_any = sum(1 for r in per if r.get("any_detected") == "1")
        row["any_detected_pct"] = f"{100 * det_any / max(n, 1):.2f}"
        for reg in REGIONS:
            det = sum(1 for r in per if r.get(f"detected_{reg}") == "1")
            row[f"det_pct_{reg}"] = f"{100 * det / max(n, 1):.2f}"
            v_med, v_q1, v_q3 = _median_iqr(_ints(per, f"vol_{reg}"))
            c_med, c_q1, c_q3 = _median_iqr(_ints(per, f"cc_count_{reg}"))
            d_med, d_q1, d_q3 = _median_iqr(_floats(per, f"disp_{reg}"))
            l_med, _, _ = _median_iqr(_floats(per, f"largest_cc_frac_{reg}"))
            row[f"vol_med_{reg}"], row[f"vol_q1_{reg}"], row[f"vol_q3_{reg}"] = f"{v_med:.0f}", f"{v_q1:.0f}", f"{v_q3:.0f}"
            row[f"cc_med_{reg}"],  row[f"cc_q1_{reg}"],  row[f"cc_q3_{reg}"]  = f"{c_med:.0f}", f"{c_q1:.0f}", f"{c_q3:.0f}"
            row[f"disp_med_{reg}"], row[f"disp_q1_{reg}"], row[f"disp_q3_{reg}"] = f"{d_med:.2f}", f"{d_q1:.2f}", f"{d_q3:.2f}"
            row[f"largest_cc_frac_med_{reg}"] = "" if math.isnan(l_med) else f"{l_med:.3f}"
        rows_out.append(row)

    TM_OUT.parent.mkdir(parents=True, exist_ok=True)
    with TM_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)
    return len(rows_out)


def harvest_seg_shift() -> int:
    if not SEG_IN.exists():
        return 0
    with SEG_IN.open() as f:
        rows = list(csv.DictReader(f))
    out = {"n_cases": len(rows)}
    for reg in REGIONS:
        nv = _floats(rows, f"native_{reg}")
        pv = _floats(rows, f"pooled_{reg}")
        out[f"native_mean_{reg}"] = f"{_mean(nv):.4f}"
        out[f"pooled_mean_{reg}"] = f"{_mean(pv):.4f}"
        out[f"delta_{reg}"] = f"{_mean(pv) - _mean(nv):+.4f}"
        out[f"native_sd_{reg}"] = f"{st.pstdev(nv):.4f}" if len(nv) > 1 else ""
        out[f"pooled_sd_{reg}"] = f"{st.pstdev(pv):.4f}" if len(pv) > 1 else ""
    fields = ["n_cases"] + [k for reg in REGIONS for k in
                            (f"native_mean_{reg}", f"native_sd_{reg}",
                             f"pooled_mean_{reg}", f"pooled_sd_{reg}",
                             f"delta_{reg}")]
    with SEG_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(out)
    return 1


if __name__ == "__main__":
    n_tm = harvest_tumor_mode()
    n_sg = harvest_seg_shift()
    print(f"[3b harvest] {TM_OUT.name}: {n_tm} sources")
    print(f"[3a harvest] {SEG_OUT.name}: {n_sg} row")
