"""SR-T2P (paper Tab. 3, retrieval rows): stitched dictionary signs -> in-betweening, scored by DTW and back-translation.

Input: the upstream SignSparK_Retrieval `stitch` output (unmodified code, ../ver_03_gf_t2p/third_party/
SignSparK_Retrieval @ b1ed927), run with the paper's own T2G predictions, `--mode train` and `--no-embed`:

    PYTHONPATH=<SignSparK_Retrieval> $PY -m signspark_retrieval stitch --dataset PHOENIX-2014T --mode train \
        --glosses checkpoints/sr_t2p/predictions/PHOENIX-2014T/test_pres.txt --no-embed \
        --annotations checkpoints/sr_t2p/annotations --data-root data --out outputs/sr_t2p/PHOENIX-2014T

Variants, all on the same stitched sequence (truncated to 304 frames) and the same test sentences:
  stitched     : upstream placeholders (cosine-eased linear blend in rot6D, 7 frames between signs);
  slerp_dense  : per-joint SLERP over every sign frame (only the transitions are filled);
  slerp_sparse : per-joint SLERP between first/middle/last of each sign;
  ssk_dense / ssk_sparse : released SignSparK KF2P with the same keyframes + the sentence (upstream `generate`
                 policies), kept frames pasted back exactly as upstream `merge` does.
Upstream `generate` samples with 50 ODE steps; --steps sets ours (paper: 10). Scores: SOKE DTW vs the real
recording, and SignSparK-BT (the dataset's model) BLEU / chrF / ROUGE, with GT through the same BT as reference.

    python eval/sr_t2p.py --dataset PHOENIX-2014T [--n 200] [--steps 10] [--device cpu]
"""
import argparse
import io
import json
import pickle
import sys
from pathlib import Path

import lmdb
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO.parent / "ver_03_gf_t2p"))
sys.path.insert(1, str(REPO / "eval"))
import bt_common as bc  # noqa: E402
import evaluate as ev  # noqa: E402
import torch  # noqa: E402
from signspark.pose_datasets_lmdb import PoseDataset_Lmdb  # noqa: E402

VARIANTS = ("stitched", "slerp_dense", "slerp_sparse", "ssk_dense", "ssk_sparse")


def read_stitched(path, ids):
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False)
    out = []
    with env.begin() as t:
        for cid in ids:
            z = np.load(io.BytesIO(t.get(cid.encode())), allow_pickle=True)
            seg = z["segment"]
            L = min(len(seg), ev.T)
            sparse = [f for b in PoseDataset_Lmdb._segment_bounds(seg[:L].tolist()) for f in (b[0], (b[0] + b[1]) // 2, b[1])]
            out.append({"id": cid, "text": f"<{z['language'][0]}> {z['translation'][0]}", "L": L,
                        "gloss": str(z["gloss"][0]), "kf_sparse": sorted(set(sparse)),
                        "kf_dense": np.flatnonzero(seg[:L] > 0).tolist(),
                        "rh": z["right_features"][:L, :90].astype(np.float32),
                        "lh": z["left_features"][:L, :90].astype(np.float32),
                        "body": z["body_features"][:L, 66:126].astype(np.float32)})
    return out


def as_streams(clips, fill):
    """fill(x, c) -> (L, D) per stream; returns evaluate.py's (H, Bd) layout, left hand in the right frame."""
    H = np.zeros((len(clips), 2, ev.T, 90), np.float32)
    Bd = np.zeros((len(clips), ev.T, 60), np.float32)
    for k, c in enumerate(clips):
        H[k, 0] = ev.pad(fill(ev.flip_left_hand_features(c["lh"]), c))
        H[k, 1] = ev.pad(fill(c["rh"], c))
        Bd[k] = ev.pad(fill(c["body"], c))
    return H, Bd


def paste_back(H, Bd, clips, policy):
    ref_H, ref_B = as_streams(clips, lambda x, c: x)
    for k, c in enumerate(clips):
        kf = c[f"kf_{policy}"]
        H[k][:, kf], Bd[k][kf] = ref_H[k][:, kf], ref_B[k][kf]
    return H, Bd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="PHOENIX-2014T", choices=["PHOENIX-2014T", "CSL-Daily"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--ckpt-dir", default=str(REPO.parent / "ver-01" / "SignSparK" / "checkpoints"))
    a = ap.parse_args()
    a.seed, a.scale = 102, -1.0
    ev.dev = torch.device(a.device)
    run = REPO / "outputs" / "sr_t2p" / a.dataset
    out_dir = run / f"eval_s{a.steps}"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = ev.read_clips(ev.lmdb_path(REPO / "data" / "lmdb", "test", a.dataset), a.n)
    st = read_stitched(run / "lmdb" / "test" / f"{a.dataset}_stitch-train_test.lmdb", [c["id"] for c in gt])
    fk = ev.sm.SoKeJoints(flat_hand_mean=True, device=a.device)
    gt_j = [fk(*ev.smplx_in(c["body"], c["rh"], c["lh"], False)) for c in gt]

    bc.BT_DIR = REPO / "checkpoints" / "bt_all" / a.dataset
    bt = bc.BT()
    refs = [bt.ref(c["text"].split("> ", 1)[1]) for c in gt]
    res = {"args": vars(a), "n": len(gt), "mean_len_gt": float(np.mean([c["L"] for c in gt])),
           "mean_len_stitched": float(np.mean([c["L"] for c in st])),
           "mean_kf": {p: float(np.mean([len(c[f"kf_{p}"]) for c in st])) for p in ("sparse", "dense")}}
    hyps = bt.translate([bc.gt_feats(c) for c in gt])
    res["gt"] = {"bt": bt.corpus(refs, hyps), "bt_examples": hyps[:5]}
    print(f"gt           BT " + " ".join(f"{k} {v:.2f}" for k, v in res["gt"]["bt"].items()), flush=True)

    models = None
    for v in a.variants:
        if v == "stitched":
            H, Bd = as_streams(st, lambda x, c: x)
        elif v.startswith("slerp"):
            pol = v.split("_")[1]
            H, Bd = as_streams(st, lambda x, c, p=pol: ev.slerp_fill(x, c[f"kf_{p}"]))
        else:
            pol = v.split("_")[1]
            if models is None:
                models = {s: ev.load(s, f"{a.ckpt_dir}/{s}/ema_0.9999_200000.pt") for s in ("hand", "body")}
            for c in st:
                c["kf"] = c[f"kf_{pol}"]
            H, Bd = ev.run_streams(models, st, list(range(len(st))), 0, a, True)
            H, Bd = paste_back(H, Bd, st, pol)
        rows = ev.score_all(fk, gt_j, st, H, Bd)
        feats = [bc.pred_feats(Bd[k, :c["L"]], H[k, 0, :c["L"]], H[k, 1, :c["L"]]) for k, c in enumerate(st)]
        hyps = bt.translate(feats)
        res[v] = {"dtw": ev.mean(rows), "bt": bt.corpus(refs, hyps), "bt_examples": hyps[:5],
                  "sent_chrf": [bc.sent_chrf(r, h) for r, h in zip(refs, hyps)]}
        np.savez_compressed(out_dir / f"{v}_pred.npz", hand=H, body=Bd, ids=np.array([c["id"] for c in st]),
                            lengths=np.array([c["L"] for c in st]))
        print(f"{v:12} DTW " + " ".join(f"{k} {x:.2f}" for k, x in res[v]["dtw"].items()) + " | BT " +
              " ".join(f"{k} {x:.2f}" for k, x in res[v]["bt"].items()), flush=True)
        (out_dir / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    res["refs_examples"] = refs[:5]
    (out_dir / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
