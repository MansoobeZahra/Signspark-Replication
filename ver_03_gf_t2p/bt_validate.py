"""Step 1: validate SignSparK-BT as a scorer.

1. Full How2Sign test GT through OUR feature path (must equal the authors' CLI: `score --split test`).
2. Controls on the full test set: standing still (GT mean pose, held), real motion of a DIFFERENT clip,
   GT with the left hand wrongly flipped (does the BT model look at the hands?).
3. Earlier outputs: model_tests' official sample.py dumps (text-only no CFG / CFG 2.5, keyframe mode) through the
   authors' own load_sample_dump; our 200-subset run (eval_t2p.py "ours", CFG 2.5, 10 steps).

    .venv-signspark/bin/python bt_validate.py
"""

import io
import json
import pickle
import time

import lmdb
import numpy as np

import bt_common as bc
from eval_t2p import LMDB, OUT as T2P_OUT, load_subset, r6
from signspark_bt.data.sample_dumps import load_sample_dump

MT = bc.ROOT.parent / "model_tests" / "signspark" / "ckpt"
DUMPS = {   # name: (hand, body, unflip_left)  -- pred left: t2p in right frame; kf2p README config true-left
    "official_t2p_nocfg_50steps": ("hand/debug0_eulerstepsize50_can1_anchor0_samples_20261019-10:17:12/seed102_clampstep0_t2p.npy",
                                   "body/debug0_eulerstepsize50_can1_anchor0_samples_20261019-10:33:36/seed102_clampstep0_t2p.npy", True),
    "official_t2p_cfg2.5_50steps": ("hand/debug0_eulerstepsize50_can1_anchor0_samples_textscale2.5/seed102_clampstep0_t2p_cfg2.5.npy",
                                    "body/debug0_eulerstepsize50_can1_anchor0_samples_textscale2.5/seed102_clampstep0_t2p_cfg2.5.npy", True),
    "official_kf2p": ("hand/debug0_eulerstepsize50_can1_anchor0_samples/seed102_clampstep0_kf2p.npy",
                      "body/debug0_eulerstepsize50_can1_anchor0_samples/seed102_clampstep0_kf2p.npy", False),
}


def all_test():
    env = lmdb.open(str(LMDB), readonly=True, lock=False)
    with env.begin() as t:
        ids = pickle.loads(t.get(b"__meta__"))["clip_ids"]
        out = []
        for i in ids:
            z = np.load(io.BytesIO(t.get(i.encode())), allow_pickle=True)
            out.append({"id": i, "text": str(z["translation"][0]), "body": z["body_features"][:, 66:126],
                        "lh": z["left_features"][:, :90], "rh": z["right_features"][:, :90]})
    return out


def ortho(f240):
    """Re-orthonormalise a mean rot6D feature vector (40 joints x 6)."""
    return r6(f240.reshape(40, 6))[:, :2].reshape(240)


def main():
    t0 = time.perf_counter()
    bt = bc.BT()
    res = {}

    def run(name, feats, refs, keep_hyps=False):
        hyps = bt.translate(feats)
        res[name] = {"n": len(feats), **bt.corpus(refs, hyps), "distinct_hyps": len(set(hyps)),
                     "examples": [[r, h] for r, h in list(zip(refs, hyps))[:4]]}
        if keep_hyps:
            res[name]["hyps"] = hyps
        s = res[name]
        print(f"{name:34} n={s['n']:5} B1 {s['bleu1']:6.2f} B4 {s['bleu4']:5.2f} chrF {s['chrf']:6.2f} "
              f"ROUGE {s['rouge']:6.2f} distinct {s['distinct_hyps']}", flush=True)
        return hyps

    clips = all_test()
    refs = [bt.ref(c["text"]) for c in clips]
    gt = [bc.gt_feats(c) for c in clips]
    run("gt_full", gt, refs)
    mean = ortho(np.concatenate(gt).mean(0))
    run("standing_still_full", [np.repeat(mean[None], len(f), 0) for f in gt], refs)
    run("other_clip_gt_full", [gt[(k + 1) % len(gt)] for k in range(len(gt))], refs)
    run("gt_left_misflipped_full", [bc.pred_feats(f[:, :60], f[:, 60:150], f[:, 150:]) for f in gt], refs)

    for name, (h, b, unflip) in DUMPS.items():
        feats, raw, names = load_sample_dump(str(MT / b), str(MT / h), poses="pred_poses", unflip_left_hand=unflip)
        run(name, [f.numpy() for f in feats], [bt.ref(r) for r in raw])

    sub = load_subset(200)
    sref = [bt.ref(c["text"]) for c in sub]
    run("gt_sub200", [bc.gt_feats(c) for c in sub], sref)
    for name in ("ours", "static_mean", "notag", "len304"):
        P = np.load(T2P_OUT / f"{name}_pred.npz")
        assert list(P["ids"]) == [c["id"] for c in sub]
        L = P["lengths"]
        run(f"{name}_sub200", [bc.pred_feats(P["body"][k, 0, :L[k]], P["hand"][k, 1, :L[k]], P["hand"][k, 0, :L[k]])
                               for k in range(len(sub))], sref)

    res["seconds"] = time.perf_counter() - t0
    (bc.ROOT / "outputs" / "bt" / "validate.json").write_text(json.dumps(res, indent=1))
    print(f"done in {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
