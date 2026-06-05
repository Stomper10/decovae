#!/usr/bin/env python3
"""Phase-2 preprocessing validation for OASIS-3 (external test cohort).
Same pipeline as validate_preproc_phase2.py: antspynet skull-strip (contrast-aware)
-> N4 (ANTs) -> RAS -> 1mm -> 192^3 crop/pad -> percentile[0,99.5]->[0,1].
Adds a cohort-wide spacing/shape survey per modality because OASIS FLAIR is
clinically 2D (thick-slice, anisotropic) -> resample blur risk worth quantifying.
"""
import csv, os, glob, tempfile
import numpy as np
import nibabel as nib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ants, antspynet
from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd,
    Orientationd, Spacingd, ResizeWithPadOrCropd, ScaleIntensityRangePercentilesd)

ROOT="/data/wonyoungjang/OASIS-3/data"
OUT="/data/wonyoungjang/decovae/preproc_validation"
os.makedirs(OUT, exist_ok=True)

# (cell -> (glob pattern, antspynet modality)). prefer primary scan (skip run-02 dups).
CELLS={
 "oasis_T1":   ("*T1w*",   "t1"),
 "oasis_T2":   ("*T2w*",   "t2"),
 "oasis_FLAIR":("*FLAIR*", "flair"),
}

def list_mod(pat):
    fs=[f for f in glob.glob(f"{ROOT}/**/{pat}.nii.gz",recursive=True) if "run-02" not in f]
    return sorted(fs)

geom=Compose([
    LoadImaged(keys="image", image_only=False),
    EnsureChannelFirstd(keys="image"),
    Orientationd(keys="image", axcodes="RAS"),
    Spacingd(keys="image", pixdim=(1.0,1.0,1.0), mode="bilinear"),
    ResizeWithPadOrCropd(keys="image", spatial_size=(192,192,192)),
    ScaleIntensityRangePercentilesd(keys="image", lower=0.0, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
])

# ---- 1) cohort-wide spacing/shape survey (cheap, all files) ----
print("=== cohort spacing survey ===")
survey=[]
for cell,(pat,mod) in CELLS.items():
    fs=list_mod(pat)
    zs=[]; thick=[]
    for f in fs:
        h=nib.load(f).header; z=h.get_zooms()[:3]
        zs.append(z); thick.append(max(z))
    zs=np.array(zs); thick=np.array(thick)
    n=len(fs)
    iso = int((thick<=1.5).sum())  # ~isotropic count
    print(f"{cell:12s} n={n:4d}  median_zoom={np.median(zs,0).round(2)}  "
          f"max-dim(slice-thick) median={np.median(thick):.2f} range[{thick.min():.2f},{thick.max():.2f}]  "
          f"iso(<=1.5mm)={iso}/{n} ({100*iso/n:.0f}%)")
    survey.append([cell,n,str(np.median(zs,0).round(2).tolist()),f"{np.median(thick):.2f}",
                   f"{thick.min():.2f}",f"{thick.max():.2f}",f"{iso}/{n}"])

# ---- 2) full-pipeline montage on one representative per modality ----
print("\n=== full pipeline (strip+N4+192^3) on 1 sample/modality ===")
rows=[]; panels=[]
for cell,(pat,mod) in CELLS.items():
    fs=list_mod(pat)
    src=fs[len(fs)//2]  # middle sample
    img=ants.image_read(src)
    prob=antspynet.brain_extraction(img, modality=mod, verbose=False)
    mask=ants.threshold_image(prob, 0.5, 1.0, 1, 0)
    brain=img*mask
    n4=ants.n4_bias_field_correction(brain, mask=mask)
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tf:
        tmp=tf.name
    ants.image_write(n4, tmp)
    out=geom({"image":tmp})["image"]
    arr=out[0].cpu().numpy() if hasattr(out,"cpu") else np.asarray(out[0])
    os.remove(tmp)
    nz=float((arr>1e-6).mean())*100
    brain_frac=float((mask.numpy()>0).mean())*100
    rows.append([cell, mod, "x".join(map(str,img.shape)), "x".join(map(str,arr.shape)),
                 f"{arr.min():.3f}/{arr.max():.3f}/{arr.mean():.3f}", f"{nz:.1f}%", f"{brain_frac:.1f}%",
                 os.path.basename(src)])
    raw=img.numpy(); rz=raw[:,:,raw.shape[2]//2]
    oz=arr[:,:,arr.shape[2]//2]
    panels.append((cell, rz, oz))
    print(f"{cell}: native {img.shape} -> {arr.shape}, %nz {nz:.1f}, brain_frac {brain_frac:.1f}")

fig,axes=plt.subplots(len(panels),2,figsize=(5,len(panels)*2.3))
for ri,(cell,rz,oz) in enumerate(panels):
    axes[ri,0].imshow(np.rot90(rz),cmap="gray"); axes[ri,0].axis("off")
    axes[ri,1].imshow(np.rot90(oz),cmap="gray",vmin=0,vmax=1); axes[ri,1].axis("off")
    axes[ri,0].set_ylabel(cell,fontsize=9,rotation=0,labelpad=34,va="center")
    if ri==0:
        axes[ri,0].set_title("RAW (input)",fontsize=10); axes[ri,1].set_title("strip+N4+192^3",fontsize=10)
plt.tight_layout(); plt.savefig(f"{OUT}/montage_oasis_strip.png",dpi=95,bbox_inches="tight")
print(f"montage: {OUT}/montage_oasis_strip.png")

with open(f"{OUT}/stats_oasis.csv","w",newline="") as f:
    w=csv.writer(f,lineterminator="\n")
    w.writerow(["cell","modality","native","out","min/max/mean","%nonzero","brain_frac","sample"])
    w.writerows(rows)
with open(f"{OUT}/spacing_survey_oasis.csv","w",newline="") as f:
    w=csv.writer(f,lineterminator="\n")
    w.writerow(["cell","n","median_zoom","median_slicethick","min_thick","max_thick","iso<=1.5mm"])
    w.writerows(survey)
print("\n"+"\n".join("  ".join(r) for r in rows))
print("=== OASIS_DONE ===")
