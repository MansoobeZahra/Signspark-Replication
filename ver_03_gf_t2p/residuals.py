"""Step 3: generation-time "residual" variants of text-only SignSparK that need no retraining.

Same 50 test sentences, lengths and seeds as bt_bestofn.py (candidates j = 0..7, seed 1000*i + j), so every variant
is paired with the step-2 baseline (CFG 2.5, 10 steps; candidates 0..7 of outputs/bestofn/cands.npz).
e = M-CLIP sentence embedding, e_mean = mean embedding of the 200 subset sentences ("the average sentence").

  cfg1, cfg5, cfg7.5   guidance scale (1 = no guidance)
  neg_mean_s{2.5,5}    negative-prompt guidance: v = v(e_mean) + s (v(e) - v(e_mean)) instead of v(null)
  emb_res_l{2,4}       embedding residual: condition on e_mean + l (e - e_mean), CFG 2.5
  mean_text            the "average sign": condition on e_mean, same noise (needed by out_res)
  out_res_l{2,3}       output residual: x_mean + l (x - x_mean), x_mean = mean_text output with the same noise
  tau{0.7,1.3}         initial noise scaled by tau
  chunks3              per-chunk text over time windows (3 word chunks, Composite), CFG 2.5

    .venv-signspark/bin/python residuals.py gen cfg1 cfg5 ...   # each variant ~1 min; outputs/residuals/<v>.npz
    .venv-signspark/bin/python residuals.py score                # outputs/residuals/results.json
"""

import json
import sys
import time

import numpy as np
import torch

import bt_common as bc
from bt_bestofn import NC, OUTD as BOD, TAG, TEST_IDX, sentences

OUTD = bc.ROOT / "outputs" / "residuals"
NV = 8
VARIANTS = {
    "cfg1": dict(scale=1.0), "cfg5": dict(scale=5.0), "cfg7.5": dict(scale=7.5),
    "neg_mean_s2.5": dict(scale=2.5, neg="__MEAN__"), "neg_mean_s5": dict(scale=5.0, neg="__MEAN__"),
    "emb_res_l2": dict(scale=2.5, emb_res=2.0), "emb_res_l4": dict(scale=2.5, emb_res=4.0),
    "mean_text": dict(scale=2.5, mean_text=True),
    "tau0.7": dict(scale=2.5, tau=0.7), "tau1.3": dict(scale=2.5, tau=1.3),
    "chunks3": dict(scale=2.5, chunks=3),
}
POST = {"out_res_l2": 2.0, "out_res_l3": 3.0}


def gen(names):
    import gen_common as g
    sub, test, _, _ = sentences()
    nt = len(test)
    lens = [c["L"] for c in test for _ in range(NV)]
    seeds = [1000 * i + j for i in range(nt) for j in range(NV)]
    OUTD.mkdir(parents=True, exist_ok=True)
    for stream, n_items in (("hand", 2), ("body", 1)):
        m, f = g.load(stream)
        tb = g.TextBank(m)
        tb.install()
        E = tb.embed([TAG + c["text"] for c in sub])
        tb.put("__MEAN__", E.mean(0))
        for name in names:
            v = dict(VARIANTS[name])
            t0 = time.perf_counter()
            texts = [TAG + c["text"] for c in test for _ in range(NV)]
            kw = dict(scale=v.get("scale", 2.5), neg=v.get("neg"), tau=v.get("tau", 1.0))
            if "emb_res" in v:
                e = tb.embed(texts)
                for t, ei in zip(texts, e):
                    tb.put(f"__RES{v['emb_res']}__{t}", tb.bank["__MEAN__"] + v["emb_res"] * (ei - tb.bank["__MEAN__"]))
                texts = [f"__RES{v['emb_res']}__{t}" for t in texts]
            if v.get("mean_text"):
                texts = ["__MEAN__"] * len(texts)
            if "chunks" in v:
                cw = [g.chunk_windows(c["text"], c["L"], v["chunks"]) for c in test for _ in range(NV)]
                kw["chunks"] = [[TAG + ch for ch in c] for c, _ in cw]
                kw["wmask"] = [w for _, w in cw]
            out = g.generate(m, f, texts, lens, seeds, n_items, steps=10, **kw)
            np.save(OUTD / f"{name}_{stream}.npy", out.astype(np.float16))
            print(f"{name} {stream}: {time.perf_counter() - t0:.0f}s", flush=True)
        del m, f
        torch.cuda.empty_cache()
    for name in names:
        H, B = np.load(OUTD / f"{name}_hand.npy"), np.load(OUTD / f"{name}_body.npy")
        feats = np.empty(len(lens), dtype=object)
        for e, L in enumerate(lens):
            feats[e] = bc.pred_feats(B[e, 0, :L].astype(np.float32), H[e, 1, :L].astype(np.float32),
                                     H[e, 0, :L].astype(np.float32)).astype(np.float16)
        np.save(OUTD / f"{name}.npy", feats, allow_pickle=True)
        (OUTD / f"{name}_hand.npy").unlink()
        (OUTD / f"{name}_body.npy").unlink()


def ortho_rows(f):
    """Re-orthonormalise every rot6D block of (T, 240) features (after extrapolation)."""
    from eval_t2p import r6
    return r6(f.reshape(len(f), 40, 6))[..., :2, :].reshape(len(f), 240).astype(np.float32)


def score():
    import soke_metric as sm
    from scipy.stats import wilcoxon
    from bt_bestofn import joints
    t0 = time.perf_counter()
    sub, test, _, _ = sentences()
    nt = len(test)
    bt = bc.BT()
    refs = [bt.ref(c["text"]) for c in test]
    Z = np.load(BOD / "cands.npz", allow_pickle=True)
    C = np.load(BOD / "ctrl_wrongtext.npz", allow_pickle=True)
    pick = lambda F: [F[i * NC + j].astype(np.float32) for i in range(nt) for j in range(NV)]  # noqa: E731
    sets = {"baseline_cfg2.5": pick(Z["feats"]), "ctrl_wrong_text_cfg2.5": pick(C["feats"])}
    for name in VARIANTS:
        p = OUTD / f"{name}.npy"
        if p.exists():
            sets[name] = [f.astype(np.float32) for f in np.load(p, allow_pickle=True)]
    if "mean_text" in sets:
        for name, lam in POST.items():
            sets[name] = [ortho_rows(m + lam * (x - m)) for x, m in zip(sets["baseline_cfg2.5"], sets["mean_text"])]

    fk = sm.SoKeJoints()
    gtR = [bc.retrieval_feats(joints(fk, bc.gt_feats(c))) for c in sub]
    own = np.array(TEST_IDX)
    res, per = {}, {}
    for name, feats in sets.items():
        hyps = np.array(bt.translate(feats)).reshape(nt, NV)
        ch = np.array([[bc.sent_chrf(refs[i], hyps[i, j]) for j in range(NV)] for i in range(nt)])
        oth = np.array([[np.mean([bc.sent_chrf(refs[k], hyps[i, j]) for k in range(nt) if k != i])
                         for j in range(NV)] for i in range(nt)])
        D = bc.dist_matrix([bc.retrieval_feats(joints(fk, f)) for f in feats], gtR).reshape(nt, NV, len(sub))
        rk = np.array([[(D[i, j] < D[i, j, own[i]]).sum() + 1 for j in range(NV)] for i in range(nt)])
        sel = ch.argmax(1)
        per[name] = {"spec": (ch - oth).mean(1), "rank": rk.mean(1)}
        res[name] = {
            "bt_cand0": bt.corpus(refs, list(hyps[:, 0])),
            "bt_best_of_8": bt.corpus(refs, [hyps[i, sel[i]] for i in range(nt)]),
            "sent_chrf_own": float(ch.mean()), "sent_chrf_other_refs": float(oth.mean()),
            "specificity": float((ch - oth).mean()),
            "retrieval_all": {"top1": float((rk == 1).mean()), "top5": float((rk <= 5).mean()),
                              "median_rank": float(np.median(rk)), "mean_rank": float(rk.mean())},
            "retrieval_best_of_8_bt_median_rank": float(np.median(rk[np.arange(nt), sel])),
            "motion_energy": float(np.mean([bc.motion_energy(f) for f in feats])),
            "distinct_hyps": int(len(set(hyps.ravel()))),
            "example": [refs[0], hyps[0, 0]]}
        print(f"{name:24} spec {res[name]['specificity']:+.2f} (own {res[name]['sent_chrf_own']:.2f}) "
              f"rank {res[name]['retrieval_all']['mean_rank']:6.1f} top5 {res[name]['retrieval_all']['top5']:.3f} "
              f"best8 B1 {res[name]['bt_best_of_8']['bleu1']:.2f} motion {res[name]['motion_energy']:.4f} "
              f"[{time.perf_counter() - t0:.0f}s]", flush=True)
    b = per["baseline_cfg2.5"]
    for name in res:
        if name == "baseline_cfg2.5":
            continue
        d = per[name]
        res[name]["vs_baseline"] = {
            "p_spec_better": float(wilcoxon(d["spec"], b["spec"], alternative="greater").pvalue),
            "p_rank_better": float(wilcoxon(b["rank"], d["rank"], alternative="greater").pvalue),
            "p_spec_worse": float(wilcoxon(b["spec"], d["spec"], alternative="greater").pvalue),
            "p_rank_worse": float(wilcoxon(d["rank"], b["rank"], alternative="greater").pvalue)}
    (OUTD / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v.get("vs_baseline") for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen(sys.argv[2:])
    else:
        score()
