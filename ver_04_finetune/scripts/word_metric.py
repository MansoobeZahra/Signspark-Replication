"""Word-level identification metric (PLAN.md §2.2) and its calibration on REAL recordings.

A motion (SignSparK streams: body (T,60), right hand (T,90), left hand in model frame (T,90)) is turned into
SOKE joints (soke_metric.SoKeJoints, flat_hand_mean=True), then the retrieval features already used in
ver_03_gf_t2p/ver_03_madis (body 12 joints pelvis-relative + both hands wrist-relative, 64 frames), and compared
by length-normalised DTW to every 3D-LEX sign. Handedness-invariant: distance = min(query, mirrored query).
Correct = own gloss ranked 1 of 735.

Calibration (this script's main): query = each converted SignAvatars-WLASL clip whose gloss is also an animated
3D-LEX sign (a different signer, a different capture method) -> rank of the 3D-LEX sign of the same gloss.
    ../ver-01/.venv-signspark/bin/python scripts/word_metric.py
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ver_03_gf_t2p"))
sys.path.insert(0, str(ROOT / "ver_02"))
import soke_metric as sm  # noqa: E402
import bt_common as bc  # noqa: E402  (retrieval_feats, dist_matrix, retrieval)
from smplx_fk import rot6d_rows_to_matrix, hand_pose_axis_angle  # noqa: E402
from smplx.joint_names import JOINT_NAMES  # noqa: E402

HERE = ROOT / "ver_04_finetune"
LEX = HERE / "data" / "3dlex"
_J = None


def joints():
    global _J
    if _J is None:
        _J = sm.SoKeJoints(flat_hand_mean=True)
    return _J


def stream_joints(body60, rh90, lh90_model):
    T = len(body60)
    body = np.zeros((T, 21, 3))
    body[:, 11:21] = Rotation.from_matrix(rot6d_rows_to_matrix(body60.reshape(T, 10, 6)).reshape(-1, 3, 3)).as_rotvec().reshape(T, 10, 3)
    return joints()(body, hand_pose_axis_angle("left", lh90_model), hand_pose_axis_angle("right", rh90))


_PAIR = None


def mirror_joints(J):
    """Reflect x and swap left/right (body by SMPL-X joint names; hands swapped wholesale)."""
    global _PAIR
    if _PAIR is None:
        n = JOINT_NAMES[:144]
        _PAIR = [n.index(s.replace("left", "@").replace("right", "left").replace("@", "right")) if
                 s.replace("left", "@").replace("right", "left").replace("@", "right") in n else i for i, s in enumerate(n)]
    f = np.array([-1.0, 1.0, 1.0])
    return {"body": J["body"][:, _PAIR] * f, "lhand": J["rhand"] * f, "rhand": J["lhand"] * f}


def feats_pair(body60, rh90, lh90):
    J = stream_joints(body60, rh90, lh90)
    return bc.retrieval_feats(J), bc.retrieval_feats(mirror_joints(J))


def load_streams(d):
    return (np.load(d / "body_rot6d.npy"), np.load(d / "hand_rot6d.npy"), np.load(d / "hand_rot6d_left_modelframe.npy"))


def lex_bank():
    cache = HERE / "outputs" / "lex_bank_feats.npz"
    if cache.exists():
        z = np.load(cache)
        return list(z["glosses"]), z["feats"]
    gl = sorted(p.name for p in LEX.iterdir() if (p / "meta.json").exists())
    F = np.stack([feats_pair(*load_streams(LEX / g))[0] for g in gl]).astype(np.float32)
    np.savez(cache, glosses=np.array(gl), feats=F)
    return gl, F


def rank_queries(queries, own, gl, F):
    """queries: list of (feat, feat_mirrored); own: gloss per query -> retrieval dict over the 3D-LEX bank."""
    D = np.minimum(bc.dist_matrix([q[0] for q in queries], list(F)), bc.dist_matrix([q[1] for q in queries], list(F)))
    return D, bc.retrieval(D, [gl.index(o) for o in own])


def main():
    gl, F = lex_bank()
    ov = json.loads((HERE / "outputs" / "vocab_overlap.json").read_text())["lex_anim_and_wlasl"]
    g2i = json.loads((ROOT / "ver_03_sr_t2p/data/meta/gloss_to_ids.json").read_text())
    conv = ROOT / "ver_03_sr_t2p/data/converted"
    q, own, ids = [], [], []
    for g in ov:
        for i in g2i[g]:
            if (conv / i).exists():
                q.append(feats_pair(*load_streams(conv / i))); own.append(g); ids.append(i)
    D, r = rank_queries(q, own, gl, F)
    ranks = [int((D[k] < D[k, gl.index(o)]).sum() + 1) for k, o in enumerate(own)]
    # per gloss: best clip (an oracle over that gloss's clips) and median clip
    per = {}
    for g in ov:
        rr = [ranks[k] for k, o in enumerate(own) if o == g]
        per[g] = {"clips": len(rr), "best_rank": min(rr), "median_rank": float(np.median(rr))}
    # self-check: 3D-LEX signs against the bank (must be rank 1), and a mirrored copy (must be rank 1 too)
    selfq = [(F[k], F[k]) for k in range(0, len(gl), 25)]
    _, rs = rank_queries(selfq, [gl[k] for k in range(0, len(gl), 25)], gl, F)
    res = {"bank": len(gl), "queries": len(q), "glosses": len(ov), "wlasl_to_3dlex": r,
           "gloss_best_clip_top1": float(np.mean([v["best_rank"] == 1 for v in per.values()])),
           "gloss_best_clip_top5": float(np.mean([v["best_rank"] <= 5 for v in per.values()])),
           "self_check": rs, "per_gloss": per, "per_clip": dict(zip(ids, ranks))}
    (HERE / "outputs" / "word_metric_calibration.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_gloss", "per_clip")}, indent=1))
    print("per gloss (clips, best, median):", {g: (v["clips"], v["best_rank"], v["median_rank"]) for g, v in per.items()})


if __name__ == "__main__":
    main()
