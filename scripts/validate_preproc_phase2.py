#!/usr/bin/env python3
"""Phase-2 preprocessing validation on RAW cohorts (IXI/HCP/ADNI):
full pipeline = skull-strip (antspynet, contrast-aware) -> N4 (ANTs) ->
reorient RAS -> 1mm -> 192^3 crop/pad -> percentile[0,99.5]->[0,1].
Saves before(raw)/after(stripped) montage per cell to confirm skull removal.
"""
import csv, os, tempfile
import numpy as np
import nibabel as nib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ants, antspynet
from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd,
    Orientationd, Spacingd, ResizeWithPadOrCropd, ScaleIntensityRangePercentilesd)

CSVDIR="/data/wonyoungjang/decovae/csv_files"
OUT="/data/wonyoungjang/decovae/preproc_validation"
os.makedirs(OUT, exist_ok=True)

# raw cohorts only (UKB/BraTS already stripped). cell -> (base, antspynet modality)
CELLS={
 "ixi_T1":("/data/wonyoungjang/IXI","t1"),
 "ixi_T2":("/data/wonyoungjang/IXI","t2"),
 "hcp_T1":("/data/wonyoungjang/HCP","t1"),
 "hcp_T2":("/data/wonyoungjang/HCP","t2"),
 "adni_T1":("/data/wonyoungjang/ADNI","t1"),
 "adni_FLAIR":("/data/wonyoungjang/ADNI","flair"),
}

geom=Compose([
    LoadImaged(keys="image", image_only=False),
    EnsureChannelFirstd(keys="image"),
    Orientationd(keys="image", axcodes="RAS"),
    Spacingd(keys="image", pixdim=(1.0,1.0,1.0), mode="bilinear"),
    ResizeWithPadOrCropd(keys="image", spatial_size=(192,192,192)),
    ScaleIntensityRangePercentilesd(keys="image", lower=0.0, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
])

rows=[]; panels=[]
for cell,(base,mod) in CELLS.items():
    rec=next(csv.DictReader(open(f"{CSVDIR}/{cell}_train.csv")))
    src=os.path.join(base, rec["rel_path"])
    img=ants.image_read(src)
    # 1) skull-strip (contrast-aware)
    prob=antspynet.brain_extraction(img, modality=mod, verbose=False)
    mask=ants.threshold_image(prob, 0.5, 1.0, 1, 0)
    brain=img*mask
    # 2) N4 on brain (mask-aware)
    n4=ants.n4_bias_field_correction(brain, mask=mask)
    # 3) -> temp nii -> MONAI geometry+intensity
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tf:
        tmp=tf.name
    ants.image_write(n4, tmp)
    out=geom({"image":tmp})["image"]
    arr=out[0].cpu().numpy() if hasattr(out,"cpu") else np.asarray(out[0])
    os.remove(tmp)
    nz=float((arr>1e-6).mean())*100
    brain_frac=float((mask.numpy()>0).mean())*100
    rows.append([cell, mod, "x".join(map(str,img.shape)), "x".join(map(str,arr.shape)),
                 f"{arr.min():.3f}/{arr.max():.3f}/{arr.mean():.3f}", f"{nz:.1f}%", f"{brain_frac:.1f}%"])
    # panel: raw mid-axial vs stripped-output mid-axial
    raw=img.numpy(); rz=raw[:,:,raw.shape[2]//2]
    oz=arr[:,:,arr.shape[2]//2]
    panels.append((cell, rz, oz))
    print(f"{cell}: native {img.shape} -> {arr.shape}, %nz {nz:.1f}, brain_frac {brain_frac:.1f}")

# montage: rows=cells, col0=raw, col1=stripped+pipeline
fig,axes=plt.subplots(len(panels),2,figsize=(5,len(panels)*2.3))
for ri,(cell,rz,oz) in enumerate(panels):
    axes[ri,0].imshow(np.rot90(rz),cmap="gray"); axes[ri,0].axis("off")
    axes[ri,1].imshow(np.rot90(oz),cmap="gray",vmin=0,vmax=1); axes[ri,1].axis("off")
    axes[ri,0].set_ylabel(cell,fontsize=9,rotation=0,labelpad=34,va="center")
    if ri==0:
        axes[ri,0].set_title("RAW (input)",fontsize=10); axes[ri,1].set_title("strip+N4+192^3",fontsize=10)
plt.tight_layout(); plt.savefig(f"{OUT}/montage_phase2_strip.png",dpi=95,bbox_inches="tight")
print(f"montage: {OUT}/montage_phase2_strip.png")

with open(f"{OUT}/stats_phase2.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["cell","modality","native","out","min/max/mean","%nonzero","brain_frac"]); w.writerows(rows)
print("\n"+"\n".join("  ".join(r) for r in rows))
print("=== PHASE2_DONE ===")
