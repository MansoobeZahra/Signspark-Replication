"""Do sentences contain the real ASL sign for their words? Sign spotting with the WLASL recognizer.

For every clip: slide windows (6 lengths, stride 3) over the motion, run the recognizer ensemble on each window
(original and mirrored), keep for every sign class its best log-probability over all windows, and subtract the
clip's mean over classes (removes "longer clip -> more windows -> higher max"). For every word w that is a
recognizer class and occurs in >= MIN_POS sentences of the source: AUC of the score of class w between clips whose
sentence contains w and clips whose sentence does not. 0.5 = the motion carries no trace of that sign.

The test is run on REAL signing first (h2s_gt: SignSparK's fits; soke_gt: MaDiS's fits). Only if real signing
scores clearly above 0.5 is the judge able to say anything about a model's output.

    python judge/spot.py            # -> outputs/judge/spot.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D  # noqa: E402
import recognizer as R  # noqa: E402

LENS = (12, 18, 24, 32, 44, 60)
STRIDE = 3
MIN_POS = 10
SKIP = {"a", "and"}      # usually not signed as a separate sign in ASL


@torch.no_grad()
def clip_scores(nets, f):
    wins = []
    for L in LENS:
        if len(f) < L:
            continue
        for s in range(0, len(f) - L + 1, STRIDE):
            wins.append(f[s:s + L])
    if not wins:
        wins = [f]
    x = torch.tensor(np.stack([R.resample(w) for w in wins]))
    xm = torch.tensor(np.stack([R.resample(D.mirror(w)) for w in wins]))
    lp = torch.stack([torch.maximum(F.log_softmax(n(x), -1), F.log_softmax(n(xm), -1)) for n in nets]).mean(0)
    best = lp.max(0).values.numpy()                         # (C,) best window per class
    return best - best.mean()


def score_source(nets, clips, name):
    cache = D.CACHE / f"spot_{name}.npy"
    if cache.exists():
        return np.load(cache)
    S = np.stack([clip_scores(nets, c["f"]) for c in clips])
    np.save(cache, S)
    return S


def evaluate(S, clips, classes, words_used=None):
    W = [D.words(c["text"]) for c in clips]
    per = {}
    for ci, w in enumerate(classes):
        if w in SKIP:
            continue
        pos = np.array([w in ws for ws in W])
        if (words_used is None and pos.sum() < MIN_POS) or (words_used is not None and w not in words_used):
            continue
        u = mannwhitneyu(S[pos, ci], S[~pos, ci], alternative="greater")
        per[w] = {"n_pos": int(pos.sum()), "auc": float(u.statistic / (pos.sum() * (~pos).sum())), "p": float(u.pvalue)}
    aucs = np.array([v["auc"] for v in per.values()])
    # pooled: every (clip, word) pair, rank of the word's class among all classes in that clip
    ranks_pos, ranks_neg = [], []
    for k, ws in enumerate(W):
        order = (-S[k]).argsort().argsort() + 1               # rank of each class in this clip
        for w in per:
            (ranks_pos if w in ws else ranks_neg).append(order[classes.index(w)])
    return {"words": len(per), "mean_auc": float(aucs.mean()) if len(aucs) else None,
            "words_auc_gt_0.5_p05": int(sum(v["p"] < 0.05 for v in per.values())),
            "pos_top5_rate": float(np.mean(np.array(ranks_pos) <= 5)) if ranks_pos else None,
            "neg_top5_rate": float(np.mean(np.array(ranks_neg) <= 5)) if ranks_neg else None,
            "per_word": per}


def main():
    ck = torch.load(D.CACHE / "recognizer.pt", weights_only=False)
    classes = ck["classes"]
    nets = []
    for sd in ck["state"]:
        n = R.Net(len(classes))
        n.load_state_dict(sd)
        nets.append(n.eval())
    soke_gt, madis = D.load_madis()
    sources = {"h2s_gt": D.load_h2s_gt(), "signspark_t2p": D.load_signspark_t2p(), "soke_gt": soke_gt, "madis": madis}
    res = {}
    for pair in (("h2s_gt", "signspark_t2p"), ("soke_gt", "madis")):
        gt_name = pair[0]
        S_gt = score_source(nets, sources[gt_name], gt_name)
        res[gt_name] = evaluate(S_gt, sources[gt_name], classes)
        used = set(res[gt_name]["per_word"])                     # same words for the model as for its real signing
        for name in pair[1:]:
            S = score_source(nets, sources[name], name)
            res[name] = evaluate(S, sources[name], classes, words_used=used)
        print(f"\n{'word':12}" + "".join(f"{n:>16}" for n in pair))
        for w in sorted(used):
            print(f"{w:12}" + "".join(f"{res[n]['per_word'][w]['auc']:10.2f} (n={res[n]['per_word'][w]['n_pos']:3d})"
                                      for n in pair))
        for n in pair:
            r = res[n]
            print(f"{n:14} mean AUC {r['mean_auc']:.3f} | words p<0.05: {r['words_auc_gt_0.5_p05']}/{r['words']} | "
                  f"word's sign in clip's top-5: {r['pos_top5_rate']:.3f} if the sentence has the word, "
                  f"{r['neg_top5_rate']:.3f} if not")
    (D.CACHE / "spot.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
