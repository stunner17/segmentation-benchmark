"""Prepare the Kromp et al. 2020 dataset (BioStudies S-BSST265) for benchmarking.

Expects the unzipped dataset in data/kromp/. Writes single-channel uint8 images and
ground-truth label masks in two versions:
  native/  as published
  scaled/  rescaled so the median ground-truth nucleus diameter is TARGET_DIAM px
           (an oracle scale: ground truth picks the scale, standing in for a known pixel size)
"""
from pathlib import Path
import numpy as np, pandas as pd, tifffile
from skimage.measure import regionprops
from skimage.transform import resize

ROOT = Path(__file__).resolve().parents[1]
TARGET_DIAM = 20.0

src = ROOT / "data/kromp"
out = ROOT / "data/prepared/kromp"
for sub in ["native/img", "native/gt", "scaled/img", "scaled/gt"]:
    (out / sub).mkdir(parents=True, exist_ok=True)

desc = pd.read_csv(src / "image_description.csv", sep=";")
rows = []
for _, r in desc.iterrows():
    name = r["Image_Name"]
    img = tifffile.imread(src / "rawimages" / f"{name}.tif")
    gt = tifffile.imread(src / "groundtruth" / f"{name}.tif").astype(np.int32)
    rgb = img.ndim == 3
    if rgb:
        # nuclear signal may sit in one colour channel; take the brightest per pixel
        img = img.max(axis=-1)
    img = img.astype(np.uint8)
    assert img.shape == gt.shape, name
    props = regionprops(gt)
    diam = float(np.median([p.equivalent_diameter_area for p in props])) if props else np.nan
    f = TARGET_DIAM / diam
    hs, ws = max(32, round(img.shape[0] * f)), max(32, round(img.shape[1] * f))
    img_s = resize(img, (hs, ws), order=1, preserve_range=True, anti_aliasing=f < 1).astype(np.uint8)
    gt_s = resize(gt, (hs, ws), order=0, preserve_range=True, anti_aliasing=False).astype(np.int32)
    tifffile.imwrite(out / "native/img" / f"{name}.tif", img)
    tifffile.imwrite(out / "native/gt" / f"{name}.tif", gt)
    tifffile.imwrite(out / "scaled/img" / f"{name}.tif", img_s)
    tifffile.imwrite(out / "scaled/gt" / f"{name}.tif", gt_s)
    rows.append(dict(image=name, preparation=r["Preparation"], split=r["Train-/Testset split"],
                     testset_class=r["Testset class"], magnification=r["Magnification"],
                     modality=r["Modality"], sn_class=r["S/N Class"], was_rgb=rgb,
                     h=img.shape[0], w=img.shape[1], n_nuclei=len(props),
                     median_diam_px=diam, scale_factor=f))
meta = pd.DataFrame(rows)
meta.to_csv(out / "metadata.csv", index=False)
print(meta.groupby("preparation")[["n_nuclei", "median_diam_px"]].describe().round(1))
print("total nuclei:", meta.n_nuclei.sum())
