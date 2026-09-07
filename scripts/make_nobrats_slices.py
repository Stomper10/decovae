#!/usr/bin/env python3
"""BraTS를 제외한 generation/recon 슬라이스 CSV를 만든다.

WHY. all_T2 의 32.9% (1,000 / 3,036) 가 BraTS 다. T2 는 원래 희소한데 그 3분의 1이
BraTS 이고, BraTS 생성은 셀 기준 최악이다 (brats_T2 38.8 against oasis_T2 13.4 and
ixi_T2 13.4). 그래서 all_T2 의 pooled FID 가 나쁜 것이 T2 자체의 어려움인지 BraTS
3분의 1 때문인지 현재로선 구분되지 않는다. all_T1 (3.8%) 과 all_FLAIR (4.5%) 는
비중이 작아 영향이 제한적이다.

이 제외는 편의가 아니라 원칙이다. 현재의 BraTS 생성물은 dx=tumor 한 비트만으로 만든
것이고 ControlNet 이후 마스크 조건 생성물로 교체된다. 폐기 예정인 측정값을 pooled
평균에 섞어두는 쪽이 오히려 부정확하다.

생성물 쪽 필터는 이 스크립트가 아니라 .cond.json 의 dx == "tumor" 로 건다. dx=tumor 는
BraTS 를 유일하게 식별하므로 (다른 어느 코호트에도 tumor 가 없다) cohort 조건이 없는
B-config 생성물에서도 정확히 걸러진다.

NULL FLOOR 도 함께 만든다. all_T2_nullA/B 도 33% 가 BraTS 이므로 현재 바닥값 0.378 은
BraTS 포함 기준이고, 표본이 1,518 -> 약 1,014 로 줄면 FID 의 유한표본 편향 때문에
바닥이 올라간다. 새 FID 를 옛 바닥과 비교하면 오독한다.

Usage:  python3 scripts/make_nobrats_slices.py [--slice_dir csv_files/gfid_slices]
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice_dir", default="csv_files/gfid_slices")
    ap.add_argument("--prefix", default="all_",
                    help="Only pooled slices need this; per-cohort slices are "
                         "already single-cohort and brats_* would become empty.")
    args = ap.parse_args()

    made = 0
    for f in sorted(glob.glob(os.path.join(args.slice_dir, f"{args.prefix}*.csv"))):
        base = os.path.basename(f)
        if "_nobrats" in base:
            continue
        d = pd.read_csv(f)
        if "cohort" not in d.columns:
            print(f"  [skip] {base}: cohort 열 없음")
            continue
        n0 = len(d)
        keep = d[d["cohort"] != "brats"]
        out = f.replace(".csv", "_nobrats.csv")
        keep.to_csv(out, index=False)
        made += 1
        print(f"  {base:26s} {n0:6d} -> {len(keep):6d}  "
              f"(brats {n0 - len(keep):5d} 제외, {100*(n0-len(keep))/n0:5.1f}%)")
    print(f"\n{made}개 파일 생성. 생성물 쪽은 .cond.json 의 dx == 'tumor' 로 거를 것.")


if __name__ == "__main__":
    main()
