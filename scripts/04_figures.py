"""Figures for the segmentation benchmark. Arial, 300 dpi, PNG + SVG."""
import os
from pathlib import Path
import numpy as np, pandas as pd, tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none",
                     "axes.edgecolor": "#888888", "axes.labelcolor": "#222222",
                     "xtick.color": "#444444", "ytick.color": "#444444"})
TOOLS = os.environ.get("TOOLS", "stardist,cellpose_sam,cellpose3_nuclei,mesmer").split(",")
LABEL = {"stardist": "StarDist", "cellpose_sam": "Cellpose-SAM",
         "cellpose3_nuclei": "Cellpose 3 nuclei", "mesmer": "Mesmer"}
COL = {"stardist": "#2a78d6", "cellpose_sam": "#eb6834",
       "cellpose3_nuclei": "#1baf7a", "mesmer": "#eda100"}
SETLAB = {"kromp_native": "Native resolution", "kromp_scaled": "Scale-normalised (oracle)"}


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


summ = pd.read_csv(RES / "kromp_summary.csv")
per = pd.read_csv(RES / "kromp_per_image_metrics.csv")

# 1. Headline metrics with 95% bootstrap CI
metrics = [("f1@0.5", "F1 @ IoU 0.5"), ("pq", "Panoptic quality"), ("aji", "AJI")]
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharey=True)
for i, s in enumerate(["kromp_native", "kromp_scaled"]):
    for j, (k, lab) in enumerate(metrics):
        ax = axes[i, j]
        for sub, off, mk in [("all", -0.12, "o"), ("tissue_section", 0.12, "s")]:
            g = summ[(summ.set == s) & (summ.subset == sub)].set_index("tool").loc[TOOLS]
            x = np.arange(len(TOOLS)) + off
            for xi, t in zip(x, TOOLS):
                ax.errorbar(xi, g.loc[t, k], yerr=[[g.loc[t, k] - g.loc[t, k + "_lo"]],
                            [g.loc[t, k + "_hi"] - g.loc[t, k]]], fmt=mk, ms=4.5,
                            color=COL[t], mfc=COL[t] if sub == "all" else "white",
                            ecolor=COL[t], elinewidth=1.2, capsize=0)
        ax.set_xticks(range(len(TOOLS)))
        ax.set_xticklabels([LABEL[t] for t in TOOLS], rotation=30, ha="right")
        ax.set_ylim(0, 1); ax.grid(axis="y", color="#e6e6e6", lw=0.6); ax.set_axisbelow(True)
        if i == 0: ax.set_title(lab, fontsize=9)
        if j == 0: ax.set_ylabel(SETLAB[s])
        if i == 0: ax.set_xticklabels([])
h = [plt.Line2D([], [], marker="o", ls="", color="#555", label="All 79 images"),
     plt.Line2D([], [], marker="s", ls="", color="#555", mfc="white", label="Tissue sections (12)")]
fig.legend(handles=h, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
fig.tight_layout()
save(fig, "fig1_headline_metrics")

# 2. F1 across IoU thresholds
thr = [0.5, 0.6, 0.7, 0.8, 0.9]
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
for ax, s in zip(axes, ["kromp_native", "kromp_scaled"]):
    g = per[per.set == s]
    for t in TOOLS:
        y = [g[g.tool == t][f"f1@{u}"].mean() for u in thr]
        ax.plot(thr, y, "-o", color=COL[t], lw=2, ms=4, label=LABEL[t])
    ax.set_title(SETLAB[s], fontsize=9); ax.set_xlabel("IoU threshold")
    ax.set_ylim(0, 1); ax.grid(color="#e6e6e6", lw=0.6); ax.set_axisbelow(True)
axes[0].set_ylabel("Mean F1")
axes[1].legend(frameon=False, loc="lower left")
fig.tight_layout()
save(fig, "fig2_f1_vs_iou")

# 3. F1@0.5 by preparation type (heatmap)
bp = pd.read_csv(RES / "kromp_by_preparation.csv")
PREP_ORDER = ["tissue section", "BM cytospin", "Tumor touch imprint", "cellline cytospin", "cellline grown"]
PREP_LAB = {"tissue section": "Tissue section", "BM cytospin": "Bone marrow cytospin",
            "Tumor touch imprint": "Tumour touch imprint", "cellline cytospin": "Cell line cytospin",
            "cellline grown": "Cell line, grown on slide"}
n = per[(per.set == "kromp_native") & (per.tool == TOOLS[0])].groupby("preparation").size()
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True, layout="constrained")
for ax, s in zip(axes, ["kromp_native", "kromp_scaled"]):
    w = bp[bp.set == s].pivot(index="preparation", columns="tool", values="f1@0.5").loc[PREP_ORDER, TOOLS]
    im = ax.imshow(w.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for (r, c), v in np.ndenumerate(w.values):
        ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7,
                color="white" if v > 0.6 else "#222")
    # thin white gaps between cells
    ax.set_xticks(np.arange(-0.5, len(TOOLS)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(PREP_ORDER)), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="both", length=0)
    ax.set_xticks(range(len(TOOLS)))
    ax.set_xticklabels([LABEL[t] for t in TOOLS], rotation=30, ha="right")
    ax.set_title(SETLAB[s], fontsize=9)
    for sp in ax.spines.values(): sp.set_visible(False)
axes[0].set_yticks(range(len(PREP_ORDER)))
axes[0].set_yticklabels([f"{PREP_LAB[p]} (n={n[p]})" for p in PREP_ORDER])
cb = fig.colorbar(im, ax=axes, shrink=0.85, aspect=25, pad=0.02)
cb.set_label("F1 @ IoU 0.5"); cb.outline.set_visible(False); cb.ax.tick_params(length=0)
save(fig, "fig3_by_preparation")

# 4. Example overlays on tissue-section images, native resolution
meta = pd.read_csv(ROOT / "data/prepared/kromp/metadata.csv")
ex = meta[meta.preparation == "tissue section"].sort_values("n_nuclei", ascending=False).image.iloc[[0, 3]]
fig, axes = plt.subplots(len(ex), len(TOOLS) + 1, figsize=(7.2, 1.6 * len(ex) + 0.3))
for r, name in enumerate(ex):
    img = tifffile.imread(ROOT / f"data/prepared/kromp/native/img/{name}.tif").astype(float)
    gt = tifffile.imread(ROOT / f"data/prepared/kromp/native/gt/{name}.tif")
    h, w = img.shape; cs = min(300, h, w); y0, x0 = (h - cs) // 2, (w - cs) // 2
    sl = np.s_[y0:y0 + cs, x0:x0 + cs]
    lo, hi = np.percentile(img[sl], [1, 99.5])
    base = np.clip((img[sl] - lo) / (hi - lo + 1e-9), 0, 1)
    labs = [("Ground truth", gt, "#ffffff")] + [
        (LABEL[t], tifffile.imread(RES / f"masks/kromp_native/{t}/{name}.tif"), COL[t]) for t in TOOLS]
    for c, (lab, m, col) in enumerate(labs):
        ax = axes[r, c]
        rgb = np.dstack([base] * 3)
        b = find_boundaries(m[sl], mode="inner")
        rgb[b] = matplotlib.colors.to_rgb(col)
        ax.imshow(rgb); ax.set_xticks([]); ax.set_yticks([])
        if r == 0: ax.set_title(lab, fontsize=8)
        if c > 0:
            f1 = per[(per.set == "kromp_native") & (per.tool == TOOLS[c - 1]) & (per.image == name)]["f1@0.5"].iloc[0]
            ax.text(4, cs - 6, f"F1 {f1:.2f}", color="white", fontsize=7, va="bottom",
                    bbox=dict(fc="black", alpha=0.5, lw=0, pad=1))
        if c == 0: ax.set_ylabel(name, fontsize=7)
fig.tight_layout()
save(fig, "fig4_tissue_examples")

# 5. Runtime
tm = pd.concat([pd.read_csv(p) for p in RES.glob("timing_*.csv")])
tm = tm[tm.set == "kromp_native"].drop_duplicates(["tool", "image"], keep="last")
tm["s_per_mp"] = tm.seconds / tm.megapixels
fig, ax = plt.subplots(figsize=(3.4, 2.4))
med = tm.groupby("tool").s_per_mp.median().reindex(TOOLS)
ax.bar(range(len(TOOLS)), med.values, color=[COL[t] for t in TOOLS], width=0.6)
for i, v in enumerate(med.values): ax.text(i, v * 1.15, f"{v:.1f}", ha="center", fontsize=7)
ax.set_yscale("log"); ax.set_ylabel("Seconds per megapixel (median)")
ax.set_xticks(range(len(TOOLS))); ax.set_xticklabels([LABEL[t] for t in TOOLS], rotation=30, ha="right")
ax.set_title("Runtime, native images, CPU only", fontsize=9)
fig.tight_layout()
save(fig, "fig5_runtime")
print("figures written")
