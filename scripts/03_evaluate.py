"""Score predicted masks against the Kromp ground truth.

Per image: IoU-matched TP/FP/FN at thresholds 0.5..0.9 (Hungarian matching on IoU),
precision, recall, F1, PQ = SQ x RQ at IoU 0.5, AJI (Kumar et al. 2017),
mean IoU of matched pairs, relative count error.
Dataset level: mean over images with 95% bootstrap CI (images resampled, 2000 draws),
and paired differences between tools (bootstrap CI + Wilcoxon signed-rank, Holm-corrected),
for all images and for the tissue-section subset.
"""
from itertools import combinations
import os
from pathlib import Path
import numpy as np, pandas as pd, tifffile
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TOOLS = os.environ.get("TOOLS", "stardist,cellpose_sam,cellpose3_nuclei,mesmer").split(",")
THR = [0.5, 0.6, 0.7, 0.8, 0.9]
rng = np.random.default_rng(0)


def relabel(a):
    u, inv = np.unique(a, return_inverse=True)
    inv = inv.reshape(a.shape)
    if u[0] != 0:
        inv = inv + 1
    return inv


def overlap_stats(gt, pr):
    """IoU matrix (n_gt x n_pr), intersections and areas, labels made consecutive."""
    gt, pr = relabel(gt), relabel(pr)
    ng, npr = gt.max(), pr.max()
    inter = coo_matrix((np.ones(gt.size), (gt.ravel(), pr.ravel())),
                       shape=(ng + 1, npr + 1)).toarray()
    ag, ap = inter.sum(1), inter.sum(0)
    inter = inter[1:, 1:]
    union = ag[1:, None] + ap[None, 1:] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return iou, inter, union, ag[1:], ap[1:]


def score(gt, pr):
    iou, inter, union, ag, ap = overlap_stats(gt, pr)
    ng, npr = iou.shape
    out = dict(n_gt=ng, n_pred=npr, count_err=(npr - ng) / max(ng, 1))
    if ng and npr:
        r, c = linear_sum_assignment(-iou)
        miou = iou[r, c]
    else:
        miou = np.array([])
    for t in THR:
        m = miou >= t
        tp = int(m.sum()); fp = npr - tp; fn = ng - tp
        out[f"tp@{t}"], out[f"fp@{t}"], out[f"fn@{t}"] = tp, fp, fn
        out[f"precision@{t}"] = tp / max(tp + fp, 1)
        out[f"recall@{t}"] = tp / max(tp + fn, 1)
        out[f"f1@{t}"] = 2 * tp / max(2 * tp + fp + fn, 1)
    m = miou >= 0.5
    tp, fp, fn = int(m.sum()), npr - int(m.sum()), ng - int(m.sum())
    out["sq"] = float(miou[m].mean()) if tp else 0.0
    out["rq"] = out["f1@0.5"]
    out["pq"] = out["sq"] * out["rq"]
    out["mean_matched_iou"] = out["sq"]
    # AJI
    if ng and npr:
        best = iou.argmax(1)
        C = inter[np.arange(ng), best].sum()
        U = union[np.arange(ng), best].sum()
        used = np.zeros(npr, bool); used[best[iou[np.arange(ng), best] > 0]] = True
        U += ap[~used].sum()
        # GT with no overlap at all: their union with "best" equals own area already
        out["aji"] = C / U if U else 0.0
    else:
        out["aji"] = 0.0
    return out


def boot_ci(x, n=2000):
    x = np.asarray(x, float)
    b = rng.choice(x, (n, len(x))).mean(1)
    return np.percentile(b, [2.5, 97.5])


def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p)
    run = 0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i]); adj[i] = min(run, 1.0)
    return adj


# ---------------- scoring ----------------
meta = pd.read_csv(ROOT / "data/prepared/kromp/metadata.csv")
rows = []
for setname in ["kromp_native", "kromp_scaled"]:
    sub = "native" if setname.endswith("native") else "scaled"
    for _, r in meta.iterrows():
        gt = tifffile.imread(ROOT / f"data/prepared/kromp/{sub}/gt/{r.image}.tif")
        for tool in TOOLS:
            p = RES / "masks" / setname / tool / f"{r.image}.tif"
            pr = tifffile.imread(p)
            if sub == "scaled":
                assert pr.shape == gt.shape
            rows.append(dict(set=setname, tool=tool, image=r.image, preparation=r.preparation,
                             magnification=r.magnification, split=r.split, **score(gt, pr)))
    print("scored", setname)
per = pd.DataFrame(rows)
per.to_csv(RES / "kromp_per_image_metrics.csv", index=False)

KEYM = ["f1@0.5", "f1@0.7", "f1@0.9", "pq", "sq", "aji", "precision@0.5", "recall@0.5", "count_err"]
summ = []
for (s, tool), g in per.groupby(["set", "tool"]):
    for subset, gg in [("all", g), ("tissue_section", g[g.preparation == "tissue section"])]:
        d = dict(set=s, tool=tool, subset=subset, n_images=len(gg), n_gt_nuclei=int(gg.n_gt.sum()))
        for k in KEYM:
            lo, hi = boot_ci(gg[k])
            d[k] = gg[k].mean(); d[k + "_lo"] = lo; d[k + "_hi"] = hi
        # pooled (micro) F1 over all nuclei
        tp, fp, fn = gg["tp@0.5"].sum(), gg["fp@0.5"].sum(), gg["fn@0.5"].sum()
        d["f1@0.5_pooled"] = 2 * tp / (2 * tp + fp + fn)
        summ.append(d)
summ = pd.DataFrame(summ)
summ.to_csv(RES / "kromp_summary.csv", index=False)

# F1 across IoU thresholds, and by preparation
curve = per.groupby(["set", "tool"])[[f"f1@{t}" for t in THR]].mean().reset_index()
curve.to_csv(RES / "kromp_f1_vs_iou.csv", index=False)
byprep = per.groupby(["set", "preparation", "tool"])[["f1@0.5", "pq", "aji"]].mean().reset_index()
byprep.to_csv(RES / "kromp_by_preparation.csv", index=False)

# paired tests: every pair of tools, per set, for F1@0.5, PQ, AJI
tests = []
for s in per.set.unique():
    g = per[per.set == s]
    for metric in ["f1@0.5", "pq", "aji"]:
        wide = g.pivot(index="image", columns="tool", values=metric)
        for a, b in combinations(TOOLS, 2):
            d = (wide[a] - wide[b]).values
            lo, hi = boot_ci(d)
            p = wilcoxon(wide[a], wide[b]).pvalue if np.any(d != 0) else 1.0
            tests.append(dict(set=s, metric=metric, tool_a=a, tool_b=b, mean_diff=d.mean(),
                              ci_lo=lo, ci_hi=hi, p_wilcoxon=p))
tests = pd.DataFrame(tests)
tests["p_holm"] = np.nan
for (s, m), idx in tests.groupby(["set", "metric"]).groups.items():
    tests.loc[idx, "p_holm"] = holm(tests.loc[idx, "p_wilcoxon"].values)
tests.to_csv(RES / "kromp_paired_tests.csv", index=False)

# same paired tests on the tissue-section subset, native resolution
tis = per[(per.set == "kromp_native") & (per.preparation == "tissue section")]
ttests = []
for metric in ["f1@0.5", "pq", "aji"]:
    wide = tis.pivot(index="image", columns="tool", values=metric)
    for a, b in combinations(TOOLS, 2):
        d = (wide[a] - wide[b]).values
        lo, hi = boot_ci(d)
        ttests.append(dict(metric=metric, tool_a=a, tool_b=b, mean_diff=d.mean(), ci_lo=lo, ci_hi=hi,
                           p_wilcoxon=wilcoxon(wide[a], wide[b]).pvalue if np.any(d != 0) else 1.0))
ttests = pd.DataFrame(ttests)
ttests["p_holm"] = np.nan
for m, idx in ttests.groupby("metric").groups.items():
    ttests.loc[idx, "p_holm"] = holm(ttests.loc[idx, "p_wilcoxon"].values)
ttests.to_csv(RES / "kromp_tissue_sections_paired_tests.csv", index=False)

# ---------------- console summary ----------------
pd.set_option("display.width", 200)
cols = ["set", "subset", "tool", "n_images", "f1@0.5", "f1@0.5_lo", "f1@0.5_hi", "pq", "aji", "count_err"]
print(summ[cols].round(3).to_string(index=False))
print(tests[(tests.metric == "f1@0.5")].round(4).to_string(index=False))
