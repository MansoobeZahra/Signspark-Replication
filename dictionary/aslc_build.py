"""ASL Citizen WiLoR + NLF + FAST features -> SignSparK-format dictionary signs, one medoid per gloss.

Per clip:
  * body  : NLF SMPL-X body_pose, joints 1-21 (SignSparK keeps 12-21: neck, collars, head, shoulders, elbows,
            wrists). Wrist rotations are NLF's.
  * hands : WiLoR MANO finger poses (15 x 3x3). Verified by MANO forward kinematics against WiLoR's own 3D
            keypoints (0.5 mm): both hands are absolute rotations (flat_hand_mean=True) in RIGHT-hand convention,
            i.e. the right hand is SMPL-X right as is, and the left hand is already SignSparK's left "model frame"
            (true SMPL-X left = diag(1,-1,-1) R diag(1,-1,-1)).
            Each WiLoR detection is assigned to the NLF wrist (2D) it is nearest to, not to WiLoR's handedness
            flag; frames without a detection are filled by per-joint SLERP (held at the ends); a hand never
            detected gets the How2Sign mean hand.
  * smoothing: rot6D Gaussian (sigma 1 frame) per stream, re-orthonormalised.
  * FAST  : trimmed to [first sign onset - 3, last sign frame + 3] (whole clip if FAST finds no sign).
  * dominance: the hand whose wrist rises more is dominant; left-dominant clips are mirrored (body L/R joints
            swapped and conjugated by diag(1,-1,-1); hand streams swapped, which in model frame is the mirror).
  * 24 fps (3D-LEX / ver_02 convention), from the video fps by per-joint SLERP.
Medoid per gloss = the clip with the smallest median distance to the gloss's other clips under ver_04's word metric
(trim 0.35, DTW, handedness-invariant). Outputs (data/asl_citizen/dict/):
  clips/<GLOSS>/<stem>/{body_rot6d,hand_rot6d,hand_rot6d_left_modelframe}.npy + meta.json   (ver_02 format)
  index.json (per gloss: clips, medoid, spread), lmdb/train/ASLCitizen-dict_train.lmdb (DATA.md §2 records)

    CUDA_VISIBLE_DEVICES= .venv-sltk/bin/python dictionary/aslc_build.py
"""
import csv
import io
import json
import pickle
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (compressed h5 datasets)
import lmdb
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
AC = REPO / "data" / "asl_citizen"
FEATS, OUT = AC / "feats", AC / "dict"
FLIP = np.diag([1.0, -1.0, -1.0])
FPS = 24.0
MIRROR_BODY = {13: 14, 14: 13, 16: 17, 17: 16, 18: 19, 19: 18, 20: 21, 21: 20}   # SMPL-X joint L<->R (12-21)


# ---------------------------------------------------------------- rotations
def r6(R):
    return R[..., :2, :].reshape(*R.shape[:-3], -1).astype(np.float32)


def from6(x):
    d = x.reshape(*x.shape[:-1], -1, 6)
    a1, a2 = d[..., :3], d[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 /= np.linalg.norm(b2, axis=-1, keepdims=True)
    return np.stack([b1, b2, np.cross(b1, b2)], -2)


def fill(R, ok):
    """R (T, J, 3, 3), ok (T,) -> per-joint SLERP over the valid frames, held at the ends."""
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return None
    if len(idx) == 1:
        return np.repeat(R[idx[:1]], len(R), 0)
    t = np.clip(np.arange(len(R)), idx[0], idx[-1])
    return np.stack([Slerp(idx, Rotation.from_matrix(R[idx, j]))(t).as_matrix() for j in range(R.shape[1])], 1)


def resample(R, fps_in, fps_out=FPS):
    n = max(2, int(round(len(R) * fps_out / fps_in)))
    t = np.linspace(0, len(R) - 1, n)
    return np.stack([Slerp(np.arange(len(R)), Rotation.from_matrix(R[:, j]))(t).as_matrix() for j in range(R.shape[1])], 1)


def smooth(R, sigma=1.0):
    return from6(gaussian_filter1d(r6(R)[..., None, :].reshape(len(R), -1), sigma, axis=0, mode="nearest"))


# ---------------------------------------------------------------- one clip
def load_clip(stem, seg, mean_rh, mean_lh_model):
    with h5py.File(FEATS / "nlf" / f"{stem}.h5") as n, h5py.File(FEATS / "wilor" / f"{stem}.h5") as w:
        fps = float(n.attrs["fps"])
        nv = n["valid"][:, 0]
        body_aa = n["params/smplx/body_pose"][:, 0].reshape(-1, 21, 3)
        kp2 = n["keypoints_2d"][:, 0]                         # (T, 55, 2): 20 = left wrist, 21 = right wrist
        wv, wb, whp = w["valid"][:], w["boxes"][:], w["params/mano/hand_pose"][:]
        wk2 = w["keypoints_2d"][:]                            # (T, K, 21, 2); 0 = wrist
    T = min(len(nv), len(wv))
    body = Rotation.from_rotvec(body_aa[:T].reshape(-1, 3)).as_matrix().reshape(T, 21, 3, 3)
    body = fill(body, nv[:T])
    if body is None:
        return None, "no NLF person"
    hands = {"rh": np.zeros((T, 15, 3, 3)), "lh": np.zeros((T, 15, 3, 3))}
    ok = {"rh": np.zeros(T, bool), "lh": np.zeros(T, bool)}
    for t in range(T):
        best = {}
        for k in np.flatnonzero(wv[t]):
            d = {s: np.linalg.norm(wk2[t, k, 0] - kp2[t, j]) for s, j in (("lh", 20), ("rh", 21))}
            s = min(d, key=d.get)
            if s not in best or d[s] < best[s][0]:            # per side: the detection nearest that wrist
                best[s] = (d[s], k)
        for s, (_, k) in best.items():
            hands[s][t], ok[s][t] = whp[t, k], True
    rh = fill(hands["rh"], ok["rh"])
    lh = fill(hands["lh"], ok["lh"])                         # WiLoR left = model frame already
    rh = rh if rh is not None else np.repeat(from6(mean_rh[None]), T, 0)
    lh = lh if lh is not None else np.repeat(from6(mean_lh_model[None]), T, 0)
    # trim: union of the FAST span and the span where a wrist is raised above rest. FAST alone undershoots on
    # webcam clips whose hands rest below the frame (WiLoR sees nothing there), e.g. SORRY trimmed to 7 frames.
    sg = seg[:T] if seg is not None else np.zeros(T)
    on = np.flatnonzero(sg > 0)
    y = kp2[:T, [20, 21], 1]                                  # image y, larger = lower
    lift = (y.max(0) - y) / np.maximum(np.ptp(y, 0), 1e-6)
    act = np.flatnonzero((lift > 0.25).any(1) & (np.ptp(y, 0) > 0.1 * np.ptp(kp2[:T, :, 1])).any())
    spans = [(on[0], on[-1])] if len(on) else []
    spans += [(act[0], act[-1])] if len(act) else []
    a, b = (max(0, min(s for s, _ in spans) - 3), min(T, max(e for _, e in spans) + 4)) if spans else (0, T)
    body, rh, lh = body[a:b], rh[a:b], lh[a:b]
    # dominance from SMPL-X wrist height is not available without FK; use wrist-rotation-free proxy:
    # WiLoR detections' 2D wrist rise (image y decreases upward) per side within the trimmed span.
    rise = {s: -(kp2[a:b, j, 1].min() - kp2[a:b, j, 1].max()) for s, j in (("lh", 20), ("rh", 21))}
    mirrored = rise["lh"] > 1.2 * rise["rh"]
    cov = {s: float(ok[s][a:b].mean()) for s in ok}
    if mirrored:
        body = mirror_body(body)
        rh, lh = lh, rh                                       # model frame: mirroring = swapping the streams
        cov = {"rh": cov["lh"], "lh": cov["rh"]}
    body, rh, lh = (smooth(resample(x, fps)) for x in (body, rh, lh))
    meta = {"fps_src": fps, "frames_src": int(T), "trim": [int(a), int(b)], "fast_signs": int((sg == 2).sum()),
            "hand_frames": cov, "mirrored": bool(mirrored), "fps": FPS}   # hand_frames: after mirroring (rh = dominant)
    return {"body": body, "rh": rh, "lh_model": lh}, meta


def mirror_body(body):
    """Mirror SMPL-X body_pose (T, 21, 3, 3) about the sagittal plane: swap L/R joints, conjugate by FLIP."""
    out = FLIP @ body @ FLIP
    for i, j in MIRROR_BODY.items():
        out[:, i - 1] = FLIP @ body[:, j - 1] @ FLIP
    return out


def save_streams(d, s, meta):
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "body_rot6d.npy", r6(s["body"][:, 11:21]))
    np.save(d / "hand_rot6d.npy", r6(s["rh"]))
    np.save(d / "hand_rot6d_left_modelframe.npy", r6(s["lh_model"]))
    (d / "meta.json").write_text(json.dumps(meta))


def lmdb_record(gloss, s):
    T = len(s["body"])
    seg = np.ones(T, np.int64)
    seg[0] = 2
    buf = io.BytesIO()
    np.savez(buf, language=np.array(["American"]), translation=np.array([gloss.lower()]), gloss=np.array([gloss]),
             segment=seg, left_features=r6(FLIP @ s["lh_model"] @ FLIP), right_features=r6(s["rh"]),
             body_features=r6(s["body"]), face_features=np.zeros((T, 56), np.float32))
    return buf.getvalue()


# ---------------------------------------------------------------- medoids
def medoids(index):
    sys.path.insert(0, str(ROOT / "ver_04_finetune" / "scripts"))
    import word_metric as wm
    import word_metric_variants as wv
    wm._J = wm.sm.SoKeJoints(flat_hand_mean=True, device="cpu")
    from compare_proto import dist_matrix  # same CPU DTW as the WLASL prototype
    for g, e in index.items():
        dirs = [OUT / "clips" / g / c for c in e["clips"]]
        if len(dirs) < 2:
            e["medoid"] = e["clips"][0] if dirs else None
            continue
        J = [wm.stream_joints(*wm.load_streams(d)) for d in dirs]
        F = [wv.feats(wv.trim(x, 0.35), "full").astype(np.float32) for x in J]
        Fm = [wv.feats(wv.trim(wm.mirror_joints(x), 0.35), "full").astype(np.float32) for x in J]
        D = np.minimum(dist_matrix(F, F), dist_matrix(Fm, F))
        D = (D + D.T) / 2
        np.fill_diagonal(D, np.nan)
        med = np.nanmedian(D, 1)
        # the medoid must have a tracked dominant hand: interpolated/mean hand shapes look typical but are not signs
        cov = np.array([json.loads((d / "meta.json").read_text())["hand_frames"]["rh"] for d in dirs])
        good = cov >= 0.6 if (cov >= 0.6).sum() >= 3 else cov >= np.sort(cov)[-3]
        e["medoid"] = e["clips"][int(np.argmin(np.where(good, med, np.inf)))]
        e["spread_median"] = float(np.median(med))
        e["ranked"] = [e["clips"][i] for i in np.argsort(med)]


def main():
    sys.path.insert(0, str(REPO / "dictionary"))
    fast = torch.load(FEATS / "fast.pt", weights_only=False)
    st = np.load(REPO / "outputs" / "static_How2Sign.npz")
    gloss_of = {}
    for s in ("train", "val", "test"):
        for r in csv.DictReader(open(AC / "ASL_Citizen" / "splits" / f"{s}.csv")):
            gloss_of[Path(r["Video file"]).stem] = (r["Gloss"], r["Participant ID"], s)
    index, records, fails = {}, [], {}
    for h in sorted((FEATS / "nlf").glob("*.h5")):
        stem = h.stem
        if stem not in gloss_of or not (FEATS / "wilor" / h.name).exists():
            continue
        g, pid, split = gloss_of[stem]
        seg = fast.get(stem)
        seg = seg.numpy() if seg is not None else None
        try:
            s, meta = load_clip(stem, seg, st["rh"], st["lh_rframe"])
        except Exception as e:  # report, keep going
            s, meta = None, f"{type(e).__name__}: {e}"
        if s is None:
            fails[stem] = meta
            continue
        meta.update(gloss=g, signer=pid, split=split, video=f"{stem}.mp4", source="ASL Citizen (MSR research license)")
        save_streams(OUT / "clips" / g / stem, s, meta)
        index.setdefault(g, {"clips": []})["clips"].append(stem)
        records.append((f"{g}/{stem}", lmdb_record(g, s)))
    medoids(index)
    (OUT / "index.json").write_text(json.dumps({"glosses": index, "failed": fails}, indent=1))
    path = OUT / "lmdb" / "train" / "ASLCitizen-dict_train.lmdb"
    path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(path), map_size=1 << 34)
    with env.begin(write=True) as t:
        for k, v in records:
            t.put(k.encode(), v)
        t.put(b"__meta__", pickle.dumps({"clip_ids": [k for k, _ in records], "num_clips": len(records)}))
    print(f"{len(records)} clips, {len(index)} glosses, {len(fails)} failed -> {OUT}")
    for g, e in sorted(index.items()):
        cov = json.loads((OUT / "clips" / g / e["medoid"] / "meta.json").read_text())["hand_frames"]
        print(f"  {g:13} {len(e['clips']):3} clips  medoid {e['medoid']}  spread {e.get('spread_median', float('nan')):.3f}  hands rh {cov['rh']:.2f} lh {cov['lh']:.2f}")


if __name__ == "__main__":
    main()
