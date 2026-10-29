"""Score OFFICIAL SignSparK sample.py dumps (How2Sign test) with the official metric code (official_metric.py).

Input: the two .npy dicts sample.py writes next to the checkpoint (hand + body stream of the same mode):
  pred_poses / gt_poses (1, B, J, 304) rot6D, hand stream B = 2 x clips interleaved L0,R0,L1,R1 (sample.py),
  lengths (1, B), text (1, B), keyframes (1, B, J, 304) bool, video_names (1, B).
Conversion: SignSparK's own tools/visualize.py:build_smplx_input (zero legs/spine/root; diag(1,-1,-1) left flip
when asked). Which left-hand convention each array is in is MEASURED, not assumed (mirror test: a left hand in
true SMPL-X convention has mean ~ mirror(mean right); one in the model's right-hand frame has mean ~ mean right),
and the flip is applied accordingly -- separately for GT and prediction. The "visualize.py default" variant
(flip both, which is what tools/visualize.py does unless --no-flip-left-hand) is scored too, as evidence.
SMPL-X joints for the metric: SOKE's definition (soke_metric.SoKeJoints: face contour, SOKE betas,
flat_hand_mean=True = SignSparK's convention).

    .venv-signspark/bin/python score_signspark.py --hand <hand.npy> --body <body.npy> --name kf2p
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "ver-01" / "SignSparK" / "tools"))
import visualize as ssv  # noqa: E402  official SignSparK conversion
import official_metric as om  # noqa: E402
sm = om.sm
OUT = ROOT / "results" / "signspark"
PAPER = {"body_pa_jpe": 7.26, "hand_pa_jpe": 2.72, "body_jpe": 6.30, "hand_jpe": 11.43}   # paper, no KF


def load(path):
    d = np.load(path, allow_pickle=True).item()
    return {k: (v[0] if isinstance(v, np.ndarray) and v.ndim and k != "num_samples" else v) for k, v in d.items()}


def left_convention(lh, rh):
    """'true_left' or 'right_frame', from the mean axis-angle of the 15 joints (rot6D rows -> rotvec)."""
    aa = lambda x: Rotation.from_matrix(ssv.rot6d_to_matrix(torch.as_tensor(x.reshape(-1, 6))).numpy()  # noqa: E731
                                        ).as_rotvec().reshape(-1, 15, 3).mean(0)
    L, R = aa(lh), aa(rh)
    d_mirror = float(np.abs(L - R * np.array([1, -1, -1])).mean())
    d_same = float(np.abs(L - R).mean())
    return ("true_left" if d_mirror < d_same else "right_frame"), {"vs_mirror_right": d_mirror, "vs_right": d_same}


def to_joints(fk, body, lh, rh, flip_left):
    t = lambda x: torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32)  # noqa: E731
    kw = ssv.build_smplx_input(t(body), t(lh), t(rh), face=None, num_expression=50, flip_left_hand=flip_left)
    n = len(body)
    return fk(kw["body_pose"].numpy().reshape(n, 21, 3), kw["left_hand_pose"].numpy().reshape(n, 15, 3),
              kw["right_hand_pose"].numpy().reshape(n, 15, 3))


def summarise(rows):
    keys = [k for k in rows[0] if k not in ("name", "text", "L")]
    m = {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}
    m["n_clips_pa"] = sum("body_pa_jpe" in r for r in rows)
    if "lhand_pa_jpe" in m:
        m["hand_pa_jpe"] = (m["lhand_pa_jpe"] + m["rhand_pa_jpe"]) / 2
    m["n_clips"] = len(rows)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", required=True)
    ap.add_argument("--body", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--max", type=int, default=0, help="score only the first N clips (0 = all)")
    ap.add_argument("--pa-max", type=int, default=300, help="DTW-PA-JPE (slow body Procrustes) on the first N clips")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    H, B = load(a.hand), load(a.body)
    hp, hg = H["pred_poses"].transpose(0, 2, 1), H["gt_poses"].transpose(0, 2, 1)     # (2N, 304, 90)
    bp, bg = B["pred_poses"].transpose(0, 2, 1), B["gt_poses"].transpose(0, 2, 1)     # (N, 304, 60)
    hn, bn = list(H["video_names"]), list(B["video_names"])
    assert hn[0::2] == hn[1::2], "hand stream is not L,R interleaved"
    names = hn[0::2]
    bidx = {n: i for i, n in enumerate(bn)}
    common = [n for n in names if n in bidx]
    L_all = H["lengths"][0::2]
    # conventions, measured on all frames of all clips
    lhg = np.concatenate([hg[2 * k, :L_all[k]] for k in range(len(names))])
    rhg = np.concatenate([hg[2 * k + 1, :L_all[k]] for k in range(len(names))])
    lhp = np.concatenate([hp[2 * k, :L_all[k]] for k in range(len(names))])
    rhp = np.concatenate([hp[2 * k + 1, :L_all[k]] for k in range(len(names))])
    conv_gt, ev_gt = left_convention(lhg, rhg)
    conv_pr, ev_pr = left_convention(lhp, rhp)
    flip_gt, flip_pr = conv_gt == "right_frame", conv_pr == "right_frame"
    print(f"left hand: GT {conv_gt} {ev_gt} -> flip {flip_gt} | prediction {conv_pr} {ev_pr} -> flip {flip_pr}", flush=True)

    fk = sm.SoKeJoints()
    # static baseline: mean GT pose (per stream feature), in the GT's own convention, held for L frames
    mean_b = np.concatenate([bg[bidx[n], :L_all[k]] for k, n in enumerate(names) if n in bidx]).mean(0)
    mean_l, mean_r = lhg.mean(0), rhg.mean(0)

    rows, rows_naive, rows_static, checks = [], [], [], []
    todo = common[:a.max] if a.max else common
    t0 = time.perf_counter()
    for c, n in enumerate(todo):
        k = names.index(n)
        L = int(L_all[k])
        assert int(B["lengths"][bidx[n]]) == L
        gb, gl, gr = bg[bidx[n], :L], hg[2 * k, :L], hg[2 * k + 1, :L]
        pb, pl, pr = bp[bidx[n], :L], hp[2 * k, :L], hp[2 * k + 1, :L]
        G = to_joints(fk, gb, gl, gr, flip_gt)
        P = to_joints(fk, pb, pl, pr, flip_pr)
        row = {"name": n, "text": str(H["text"][2 * k]), "L": L, **om.dtw_jpe_official(P, G)}
        if c < a.pa_max:
            s = sm.score(P, G)
            row.update({f"{p}_pa_jpe": s[f"{p}_pa_jpe"] for p in ("body", "lhand", "rhand")})
        rows.append(row)
        # evidence: visualize.py's default (flip both) on the same clip
        rows_naive.append({"name": n, **om.dtw_jpe_official(to_joints(fk, pb, pl, pr, True), to_joints(fk, gb, gl, gr, True))})
        S = to_joints(fk, np.tile(mean_b, (L, 1)), np.tile(mean_l, (L, 1)), np.tile(mean_r, (L, 1)), flip_gt)
        srow = {"name": n, **om.dtw_jpe_official(S, G)}
        if c < a.pa_max:
            s = sm.score(S, G)
            srow.update({f"{p}_pa_jpe": s[f"{p}_pa_jpe"] for p in ("body", "lhand", "rhand")})
        rows_static.append(srow)
        if c < 3:                                    # SOKE original code cross-check on the first clips
            checks.append({"name": n, **om.crosscheck(P, G)})
        if c % 200 == 0:
            print(f"  {c}/{len(todo)} clips, {time.perf_counter() - t0:.0f}s", flush=True)
    res = {"name": a.name, "hand_npy": a.hand, "body_npy": a.body, "left_convention": {
               "gt": conv_gt, "gt_evidence": ev_gt, "pred": conv_pr, "pred_evidence": ev_pr},
           "keyframes_per_clip_mean": float(H["keyframes"][1::2].any(1).sum(-1).mean()),
           "model": summarise(rows), "visualize_default_flip_both": summarise(rows_naive),
           "static_mean_pose": summarise(rows_static), "paper_no_kf": PAPER, "soke_original_crosscheck": checks,
           "per_clip": rows}
    (OUT / f"{a.name}.json").write_text(json.dumps(res, indent=1))
    keys = ("body_jpe", "hand_jpe", "body_pa_jpe", "hand_pa_jpe")
    print(f"\n{a.name}: {len(rows)} clips, {time.perf_counter() - t0:.0f}s")
    for lab, m in (("model", res["model"]), ("visualize default", res["visualize_default_flip_both"]),
                   ("static mean pose", res["static_mean_pose"]), ("paper (no KF)", PAPER)):
        print(f"  {lab:18}" + "  ".join(f"{k} {m.get(k, float('nan')):6.2f}" for k in keys))


if __name__ == "__main__":
    main()
