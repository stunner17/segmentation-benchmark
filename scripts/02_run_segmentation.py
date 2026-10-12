"""Run one segmentation tool over every prepared image set, out of the box.

Usage:  <env python> 02_run_segmentation.py <tool> [set1,set2]
tool in: stardist | cellpose_sam | cellpose3_nuclei | mesmer

Settings are each tool's documented defaults for nuclei, so the comparison is
"what you get without tuning":
  stardist          2D_versatile_fluo, normalize 1-99.8 pct, default prob/nms thresholds
  cellpose_sam      Cellpose 4 default model (cpsam), no diameter given, float32 weights
  cellpose3_nuclei  Cellpose 3 'nuclei' model with its built-in diameter estimation
  mesmer            DeepCell Mesmer, compartment='nuclear', membrane channel left empty,
                    image_mpp=0.5 (its default)
Masks go to results/masks/<set>/<tool>/, timings to results/timing_<tool>.csv
"""
import os, sys, time
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np, pandas as pd, tifffile

ROOT = Path(__file__).resolve().parents[1]
tool = sys.argv[1]
SETS = {
    "kromp_native": ROOT / "data/prepared/kromp/native/img",
    "kromp_scaled": ROOT / "data/prepared/kromp/scaled/img",
}


def load_model():
    if tool == "stardist":
        from stardist.models import StarDist2D
        from csbdeep.utils import normalize
        m = StarDist2D.from_pretrained("2D_versatile_fluo")
        def run(img, setname):
            lab, _ = m.predict_instances(normalize(img, 1, 99.8), n_tiles=m._guess_n_tiles(img))
            return lab
    elif tool == "cellpose_sam":
        from cellpose import models
        # bfloat16 (the default) is >10x slower on CPUs without native bf16; use float32
        m = models.CellposeModel(gpu=False, use_bfloat16=False)
        def run(img, setname):
            return m.eval(img.astype(np.float32))[0]
    elif tool == "cellpose3_nuclei":
        from cellpose import models
        m = models.Cellpose(gpu=False, model_type="nuclei")
        def run(img, setname):
            return m.eval(img.astype(np.float32), diameter=None, channels=[0, 0])[0]
    elif tool == "mesmer":
        from deepcell.applications import Mesmer
        m = Mesmer()
        def run(img, setname):
            # Mesmer fails on inputs smaller than its 256 px tile: zero-pad, then crop back
            h, w = img.shape
            x = np.zeros((1, max(h, 256), max(w, 256), 2), np.float32)
            x[0, :h, :w, 0] = img
            return m.predict(x, image_mpp=0.5, compartment="nuclear")[0, :h, :w, 0]
    else:
        raise SystemExit(f"unknown tool {tool}")
    return run


if len(sys.argv) > 2:   # optional: restrict to some sets, e.g. "kromp_scaled"
    SETS = {k: v for k, v in SETS.items() if k in sys.argv[2].split(",")}
run = load_model()
rows = []
for setname, d in SETS.items():
    od = ROOT / "results/masks" / setname / tool
    od.mkdir(parents=True, exist_ok=True)
    files = sorted(d.glob("*.tif"))
    if os.environ.get("REVERSE"):   # lets a second process work the same set from the other end
        files = files[::-1]
    for i, f in enumerate(files):
        if (od / f.name).exists() and not os.environ.get("RERUN"):
            continue   # resume: keep masks already written (timing rows are lost for these)
        img = tifffile.imread(f)
        t0 = time.perf_counter()
        lab = run(img, setname)
        dt = time.perf_counter() - t0
        tifffile.imwrite(od / f.name, np.asarray(lab).astype(np.int32), compression="zlib")
        rows.append(dict(tool=tool, set=setname, image=f.stem, seconds=dt,
                         megapixels=img.size / 1e6, n_objects=int(len(np.unique(lab)) - 1)))
        print(f"[{tool}] {setname} {i + 1}/{len(files)} {f.stem} {dt:.1f}s n={rows[-1]['n_objects']}", flush=True)
        tp = ROOT / f"results/timing_{tool}.csv"
        prev = pd.read_csv(tp) if tp.exists() else pd.DataFrame()
        pd.concat([prev, pd.DataFrame(rows[-1:])]).to_csv(tp, index=False)
print("DONE", tool)
