"""Rescore saved evaluate.py KF2P generations with the GT keyframes pasted back (as SignSparK_Retrieval's `merge`
does), and report how far the model drifts from its own keyframes.

    CUDA_VISIBLE_DEVICES= python eval/rescore_pasteback.py released_PHOENIX-2014T_kf2pdense_n100 [...]
Writes outputs/eval/<run>_pasteback.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
import evaluate as ev  # noqa: E402


def main():
    fk = ev.sm.SoKeJoints(flat_hand_mean=True, device="cpu")
    for run in sys.argv[1:]:
        res = json.loads((REPO / "outputs" / "eval" / f"{run}.json").read_text())
        a = res["args"]
        z = np.load(REPO / "outputs" / "eval" / f"{run}_pred.npz")
        clips = ev.read_clips(ev.lmdb_path(a["lmdb_root"], "test", a["dataset"]), a["n"])
        assert [str(i) for i in z["ids"]] == [c["id"] for c in clips]
        H, Bd = z["hand"].copy(), z["body"].copy()
        drift = []
        for k, c in enumerate(clips):
            kf = c["kf_dense"] if a.get("keyframes") == "dense" else c["kf"]
            gl, gr, gb = ev.flip_left_hand_features(c["lh"])[kf], c["rh"][kf], c["body"][kf]
            if not kf:
                continue
            drift.append(np.abs(np.concatenate([H[k, 0, kf] - gl, H[k, 1, kf] - gr, Bd[k, kf] - gb], 1)).mean())
            H[k, 0, kf], H[k, 1, kf], Bd[k, kf] = gl, gr, gb
        gt_j = [fk(*ev.smplx_in(c["body"], c["rh"], c["lh"], False)) for c in clips]
        rows = ev.score_all(fk, gt_j, clips, H, Bd)
        out = {"run": run, "model_pasteback": ev.mean(rows), "model_raw": res["model"],
               "keyframe_drift_rot6d_mean_abs": float(np.mean(drift)),
               "per_clip": [{"id": c["id"], **{m: r[m] for m in ev.METRICS}} for c, r in zip(clips, rows)]}
        (REPO / "outputs" / "eval" / f"{run}_pasteback.json").write_text(json.dumps(out, indent=1))
        print(run, "raw", {k: round(v, 2) for k, v in res["model"].items()}, "\n   pasted",
              {k: round(v, 2) for k, v in out["model_pasteback"].items()}, "drift", round(out["keyframe_drift_rot6d_mean_abs"], 4),
              flush=True)


if __name__ == "__main__":
    main()
