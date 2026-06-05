#!/usr/bin/env python3
"""Phase-1 preprocessing smoke test: per (cohort x modality) cell, run the
geometry+intensity pipeline (reorient RAS -> 1mm spacing -> 192^3 crop/pad ->
percentile[0,99.5]->[0,1]) on a few samples. Records native->output shape +
intensity stats; saves mid-slice montages for visual QC.
NOTE: skull-strip + N4 (offline, raw cohorts) NOT included here (Phase 2).
UKB uses native _unbiased_brain (planned source); BraTS already stripped+1mm.
"""
import csv, os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd,
    Orientationd, Spacingd, ResizeWithPadOrCropd, ScaleIntensityRangePercentilesd)

CSVDIR="/data/wonyoungjang/decovae/csv_files"
OUT="/data/wonyoungjang/decovae/preproc_validation"
os.makedirs(OUT, exist_ok=True)
N=2  # samples per cell

BASE={
 "ukb_T1":"/data/wonyoungjang/20252_unzip", "ukb_FLAIR":"/data/wonyoungjang/20253_unzip",
 "ixi_T1":"/data/wonyoungjang/IXI", "ixi_T2":"/data/wonyoungjang/IXI",
 "hcp_T1":"/data/wonyoungjang/HCP", "hcp_T2":"/data/wonyoungjang/HCP",
 "brats_T1":"/data/wonyoungjang/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
 "brats_T1c":"/data/wonyoungjang/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
 "brats_T2":"/data/wonyoungjang/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
 "brats_FLAIR":"/data/wonyoungjang/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
 "adni_T1":"/data/wonyoungjang/ADNI", "adni_FLAIR":"/data/wonyoungjang/ADNI",
}
CELLS=list(BASE.keys())

def resolve(cell, rel):
    p=os.path.join(BASE[cell], rel)
    if cell=="ukb_T1":   p=p.replace("T1_brain_to_MNI","T1_unbiased_brain")
    if cell=="ukb_FLAIR":p=p.replace("T2_FLAIR_brain_to_MNI","T2_FLAIR_unbiased_brain")
    return p

tf=Compose([
    LoadImaged(keys="image", image_only=False, ensure_channel_first=False),
    EnsureChannelFirstd(keys="image"),
    Orientationd(keys="image", axcodes="RAS"),
    Spacingd(keys="image", pixdim=(1.0,1.0,1.0), mode="bilinear"),
    ResizeWithPadOrCropd(keys="image", spatial_size=(192,192,192)),
    ScaleIntensityRangePercentilesd(keys="image", lower=0.0, upper=99.5,
                                    b_min=0.0, b_max=1.0, clip=True),
])

rows_stat=[]
fig_cells=[]
for cell in CELLS:
    recs=list(csv.DictReader(open(f"{CSVDIR}/{cell}_train.csv")))[:N]
    for i,r in enumerate(recs):
        src=resolve(cell, r["rel_path"])
        if not os.path.exists(src):
            rows_stat.append([cell,i,r["rel_path"],"MISSING","","","",""]); continue
        nimg=nib.load(src)
        nat_shape="x".join(map(str,nimg.shape))
        nat_sp=",".join(f"{z:.2f}" for z in nimg.header.get_zooms()[:3])
        out=tf({"image":src})["image"]  # [1,192,192,192]
        arr=out[0].cpu().numpy() if hasattr(out,"cpu") else np.asarray(out[0])
        oshape="x".join(map(str,arr.shape))
        nz=float((arr>1e-6).mean())*100
        rows_stat.append([cell,i,os.path.basename(src),nat_shape,nat_sp,oshape,
                          f"{arr.min():.3f}/{arr.max():.3f}/{arr.mean():.3f}", f"{nz:.1f}%"])
        # montage: 3 orthogonal mid-slices
        if i==0:
            mid=[arr.shape[0]//2,arr.shape[1]//2,arr.shape[2]//2]
            fig_cells.append((cell, arr, mid))

# 셀별 대표 montage 한 장 (12 cells x 3 planes)
ncol=3; nrow=len(fig_cells)
fig,axes=plt.subplots(nrow,ncol,figsize=(ncol*2.2,nrow*2.2))
for ri,(cell,arr,mid) in enumerate(fig_cells):
    planes=[arr[mid[0],:,:], arr[:,mid[1],:], arr[:,:,mid[2]]]
    for ci,pl in enumerate(planes):
        ax=axes[ri,ci]; ax.imshow(np.rot90(pl),cmap="gray",vmin=0,vmax=1); ax.axis("off")
        if ci==0: ax.set_ylabel(cell,fontsize=8,rotation=0,labelpad=30,va="center")
        ax.set_title(["axial","coronal","sagittal"][ci] if ri==0 else "",fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/montage_all_cells.png",dpi=90,bbox_inches="tight")
print(f"montage saved: {OUT}/montage_all_cells.png")

# stats table
with open(f"{OUT}/stats.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["cell","i","file","native_shape","native_spacing","out_shape","min/max/mean","%nonzero"]); w.writerows(rows_stat)
print(f"\n{'cell':12s}{'native_shape':16s}{'native_sp':14s}{'out':12s}{'min/max/mean':22s}{'%nz':6s}")
for r in rows_stat:
    print(f"{r[0]:12s}{str(r[3]):16s}{str(r[4]):14s}{str(r[5]):12s}{str(r[6]):22s}{str(r[7]):6s}")
print("=== VALIDATE_DONE ===")
