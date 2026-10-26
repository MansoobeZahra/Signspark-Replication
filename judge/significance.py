"""Permutation test on the cached spotting scores (judge/spot.py): shuffle which sentence belongs to which clip.

Two word sets per source pair, both chosen from the REAL signing only:
  all17      : every recognizer class with >= 10 test sentences (except "a", "and");
  detectable : words whose sign the judge finds in real signing (GT AUC p < 0.05).
Statistic: mean AUC over the word set. p = share of 2,000 shuffles with a mean AUC >= the observed one.
    python judge/significance.py        # -> outputs/judge/significance.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D  # noqa: E402

N_PERM = 2000


def mean_auc(S, pos_sets, cols, perm=None):
    aucs = []
    for w, c in cols.items():
        pos = pos_sets[w] if perm is None else pos_sets[w][perm]
        r = rankdata(S[:, c])
        n1, n0 = pos.sum(), (~pos).sum()
        aucs.append((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return float(np.mean(aucs))


def main():
    spot = json.loads((D.CACHE / "spot.json").read_text())
    import torch
    classes = torch.load(D.CACHE / "recognizer.pt", weights_only=False)["classes"]
    soke_gt, madis = D.load_madis()
    texts = {"h2s_gt": [c["text"] for c in D.load_h2s_gt()], "signspark_t2p": [c["text"] for c in D.load_signspark_t2p()],
             "soke_gt": [c["text"] for c in soke_gt], "madis": [c["text"] for c in madis]}
    rng = np.random.default_rng(0)
    out = {}
    for gt, model in (("h2s_gt", "signspark_t2p"), ("soke_gt", "madis")):
        sets = {"all17": list(spot[gt]["per_word"]),
                "detectable": [w for w, v in spot[gt]["per_word"].items() if v["p"] < 0.05]}
        for src in (gt, model):
            S = np.load(D.CACHE / f"spot_{src}.npy")
            W = [D.words(t) for t in texts[src]]
            for sname, ws in sets.items():
                cols = {w: classes.index(w) for w in ws}
                pos = {w: np.array([w in x for x in W]) for w in ws}
                obs = mean_auc(S, pos, cols)
                null = np.array([mean_auc(S, pos, cols, rng.permutation(len(W))) for _ in range(N_PERM)])
                out[f"{src}/{sname}"] = {"words": ws, "mean_auc": obs, "null_mean": float(null.mean()),
                                         "null_95pct": float(np.quantile(null, 0.95)),
                                         "p": float((np.sum(null >= obs) + 1) / (N_PERM + 1))}
                print(f"{src:14} {sname:10} ({len(ws):2d} words)  mean AUC {obs:.3f}   chance {null.mean():.3f} "
                      f"(95%: {np.quantile(null, 0.95):.3f})   p = {out[f'{src}/{sname}']['p']:.4f}", flush=True)
    (D.CACHE / "significance.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
