"""ASL Citizen WiLoR + NLF + FAST features -> SignSparK-format dictionary signs, one medoid per gloss.

Per clip:
  * body  : NLF SMPL-X body_pose, joints 1-21 (SignSparK keeps 12-21: neck, collars, head, shoulders, elbows,
            wrists). Wrist rotations are NLF's. On by default: arm-depth fix (fix_arm_depth, WiLoR depth x 0.9),
            spine fold (fold_spine: NLF's spine1-3 bend folded into neck + collars, since the streams drop joints
            1-11) and the ASL-LEX contact snap (apply_contact -> contact_snap.py). See dictionary/WORKLOG.md.
  * hands : WiLoR MANO finger poses (15 x 3x3). Verified by MANO forward kinematics against WiLoR's own 3D
            keypoints (0.5 mm): both hands are absolute rotations (flat_hand_mean=True) in RIGHT-hand convention,
            i.e. the right hand is SMPL-X right as is, and the left hand is already SignSparK's left "model frame"
            (true SMPL-X left = diag(1,-1,-1) R diag(1,-1,-1)).
            Each WiLoR detection is assigned to the NLF wrist (2D) it is nearest to, not to WiLoR's handedness
            flag; frames without a detection are filled by per-joint SLERP (held at the ends); a hand never
            detected gets the How2Sign mean hand.
  * denoising, per clip and per stream (not one setting for all clips): one-frame WiLoR glitches are dropped by a
            robust z-score on the clip's own frame-to-frame rotation changes and re-filled; then a rot6D Gaussian
            whose strength is chosen from the clip's own noise estimate (adaptive_smooth), re-orthonormalised.
  * trim  : union of the FAST span and the span where a wrist is raised above rest (NLF 2D), +-3 frames.
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
import os
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
FEATS, OUT = AC / "feats", Path(os.environ.get("ASLC_DICT", AC / "dict"))   # ASLC_DICT: A/B builds
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


def drop_spikes(R, ok, z=4.0):
    """Unmark tracked frames whose pose jumps away and straight back (one-frame tracker glitches): robust z-score
    (median/MAD of this clip's own frame-to-frame geodesic angles) above z on both the jump in and the jump out."""
    idx = np.flatnonzero(ok)
    if len(idx) < 5:
        return ok, 0
    rel = np.einsum("tjab,tjac->tjbc", R[idx[:-1]], R[idx[1:]])
    ang = np.arccos(np.clip((np.trace(rel, axis1=-2, axis2=-1) - 1) / 2, -1, 1)).mean(1)
    zs = (ang - np.median(ang)) / (1.4826 * np.median(np.abs(ang - np.median(ang))) + 1e-6)
    spike = (zs[:-1] > z) & (zs[1:] > z)
    ok = ok.copy()
    ok[idx[1:-1][spike]] = False
    return ok, int(spike.sum())


SIGMAS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)


def adaptive_smooth(R, obs):
    """Per-clip, per-stream smoothing strength by the discrepancy principle: the noise level is estimated from this
    clip's own observed frames (robust second differences, var(d2) = 6 sigma_n^2 for white noise), and the
    strongest Gaussian whose residual stays within that noise level is used. Clean clips are barely touched,
    jittery ones are smoothed more. Interpolated (unobserved) frames are already smooth and are left out."""
    x = r6(R)
    v = obs[:-2] & obs[1:-1] & obs[2:]
    if len(x) < 7 or v.sum() < 4:
        return R, 0.0
    d2 = np.diff(x, 2, axis=0)[v]
    noise = 1.4826 * np.median(np.abs(d2 - np.median(d2, 0)), 0) / np.sqrt(6)       # per rot6D dim
    best = 0.0
    for sg in SIGMAS:
        res = (x - gaussian_filter1d(x, sg, axis=0, mode="nearest"))[obs]
        if np.sqrt((res ** 2).mean(0)).mean() > noise.mean():
            break
        best = sg
    return (from6(gaussian_filter1d(x, best, axis=0, mode="nearest")) if best else R), best


# ---------------------------------------------------------------- arm depth (NLF elbow over-flexion)
ARM = {"rh": (14, 17, 19, 21), "lh": (13, 16, 18, 20)}          # collar, shoulder, elbow, wrist (SMPL-X joints)
SPINE = (3, 6, 9)


def _align(u, v):
    """Smallest rotation taking direction u to direction v (both 3-vectors)."""
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    ax = np.cross(u, v)
    s, c = np.linalg.norm(ax), float(u @ v)
    if s < 1e-8:
        return np.eye(3)
    return Rotation.from_rotvec(ax / s * np.arctan2(s, c)).as_matrix()


def hand_depth(k3, k2, f, left):
    """Camera depth (m) of a WiLoR hand from its apparent size: orthographic scale of its 3D keypoints onto its 2D.
    WiLoR's left-hand 3D keypoints are in the mirrored (right-hand) frame: x is flipped back first."""
    a = k3[:, :2] * (np.array([-1.0, 1.0]) if left else 1.0)
    a = a - a.mean(0)
    c = k2 - k2.mean(0)
    s = (a * c).sum() / max((a * a).sum(), 1e-12)
    return f / abs(s) if abs(s) > 1e-6 else np.nan


def fix_arm_depth(body, go, nk3, nk2, nv, det, wk3, wk2, whand):
    """NLF from a front camera over-flexes the elbows (median elbow angle 59-61 deg on ASL Citizen and WLASL alike vs
    77 on 3D-LEX mocap; its wrist depth barely varies between signs), while WiLoR's hand depth from apparent hand
    size does (hand on chest ~0.1 m in front of the shoulder, pointing signs ~0.28 m). Per arm and frame where WiLoR
    saw that hand: the NLF wrist slides along its camera ray to WiLoR's depth (the 2D image position stays exactly as
    filmed), shoulder + elbow are re-solved by two-link IK in the original elbow swivel plane, and the wrist keeps its
    global orientation. The depth change is robustly smoothed over time and fades out over undetected frames.
    body (T, 21, 3, 3) local rotations, go (T, 3) global orient, nk3 (T, 55, 3) mm, nk2 (T, 55, 2) px,
    det[side] = list of (t, k) WiLoR detections assigned to that side. Returns corrected body, mean |dz| per side."""
    X, U = nk3[nv].reshape(-1, 3) / 1000, nk2[nv].reshape(-1, 2)
    # keypoints at depth 0 or non-finite give inf rows, on which lstsq (LAPACK) never returns (stalled one shard 50 min)
    keep = np.isfinite(X).all(1) & np.isfinite(U).all(1) & (X[:, 2] > 0.1)
    X, U = X[keep], U[keep]
    if len(X) < 10:
        return body, {"rh": 0.0, "lh": 0.0, "skipped": "no valid NLF keypoints for the camera fit"}
    A = np.zeros((2 * len(X), 3))
    A[0::2, 0], A[0::2, 1], A[1::2, 0], A[1::2, 2] = X[:, 0] / X[:, 2], 1, X[:, 1] / X[:, 2], 1
    f = np.linalg.lstsq(A, U.reshape(-1), rcond=None)[0][0]           # NLF's pinhole (fov attr is not px)
    T = len(body)
    body = body.copy()
    G0 = Rotation.from_rotvec(go).as_matrix()
    stats = {}
    for side, (jc, js, je, jw) in ARM.items():
        dz = np.full(T, np.nan)
        for t, k in det[side]:
            if nv[t]:
                dz[t] = DEPTH_SCALE * hand_depth(wk3[t, k], wk2[t, k], f, whand[t, k] == 0) - nk3[t, jw, 2] / 1000
        ok = np.isfinite(dz)
        if ok.sum() < 3:
            stats[side] = 0.0
            continue
        med = np.median(dz[ok])
        bad = ok & (np.abs(dz - med) > 3 * 1.4826 * np.median(np.abs(dz[ok] - med)) + 0.02)
        ok &= ~bad
        idx = np.flatnonzero(ok)
        full = np.interp(np.arange(T), idx, dz[idx])
        full = gaussian_filter1d(full, 2.0, mode="nearest")
        gap = np.min(np.abs(np.arange(T)[:, None] - idx[None]), 1)
        full = np.clip(full * np.exp(-gap / 4.0), -0.25, 0.25)
        for t in np.flatnonzero(nv):
            S, E, W = (nk3[t, j] / 1000 for j in (js, je, jw))
            if abs(full[t]) < 1e-3 or not np.isfinite(W).all() or W[2] <= 0.1:
                continue
            Wt = W * (W[2] + full[t]) / W[2]                         # same camera ray, new depth
            a, b = np.linalg.norm(E - S), np.linalg.norm(W - E)
            d = np.clip(np.linalg.norm(Wt - S), abs(a - b) + 1e-3, 0.995 * (a + b))
            u = (Wt - S) / np.linalg.norm(Wt - S)
            Wt = S + d * u
            x = (a * a - b * b + d * d) / (2 * d)
            r = (E - S) - ((E - S) @ u) * u
            if np.linalg.norm(r) < 1e-6:
                continue
            Et = S + x * u + np.sqrt(max(a * a - x * x, 0)) * r / np.linalg.norm(r)
            # global rotations along the chain pelvis -> spine1-3 -> collar -> shoulder -> elbow -> wrist
            Gp = G0[t]
            for j in (*SPINE, jc):
                Gp = Gp @ body[t, j - 1]
            Gs = Gp @ body[t, js - 1]
            Ge = Gs @ body[t, je - 1]
            Gw = Ge @ body[t, jw - 1]
            R1 = _align(E - S, Et - S)
            Gs2 = R1 @ Gs
            R2 = _align(R1 @ (W - E), Wt - Et)
            Ge2 = R2 @ R1 @ Ge
            body[t, js - 1] = Gp.T @ Gs2
            body[t, je - 1] = Gs2.T @ Ge2
            body[t, jw - 1] = Ge2.T @ Gw
        stats[side] = float(np.abs(full[nv]).mean())
    return body, stats


def _pitch(R):
    """Forward pitch (deg) of rotations' y axis (y up, z toward the viewer)."""
    return np.degrees(np.arctan2(R[..., 2, 1], R[..., 1, 1]))


def fold_spine(body, alpha=None, head_pitch=0.0):
    """SignSparK streams keep joints 12-21 only, so NLF's spine1-3 bend (median ~+24 deg forward relative to the
    pelvis on webcam clips) is rendered straight and the arms, attached to spine3, swing up with it (hands at the
    chin). Folding the spine rotation into neck and collars keeps everything above spine3 where NLF put it
    relative to the pelvis."""
    alpha = FOLD_ALPHA if alpha is None else alpha
    S = body[:, 2] @ body[:, 5] @ body[:, 8]
    if alpha != 1.0:
        S = Rotation.from_rotvec(Rotation.from_matrix(S).as_rotvec() * alpha).as_matrix()
    body = body.copy()
    for j in (12, 13, 14):
        body[:, j - 1] = S @ body[:, j - 1]
    # the head: folding into the neck makes the head inherit NLF's pelvis tilt (the avatar looked down at its chest,
    # +32 deg); load_clip passes head_pitch = (head pitch in the camera, as filmed) - (rendered head pitch), so the
    # avatar's head faces the viewer the way the signer faces the camera.
    if head_pitch:
        body[:, 11] = Rotation.from_euler("x", head_pitch, degrees=True).as_matrix() @ body[:, 11]
    return body


# ---------------------------------------------------------------- one clip
ARM_DEPTH = True
FOLD_ALPHA = float(os.environ.get("ASLC_FOLD_ALPHA", "1.0"))       # fraction of the spine bend folded in
DEPTH_SCALE = float(os.environ.get("ASLC_DEPTH_SCALE", "0.9"))     # x WiLoR hand depth (<1: hands toward camera);
# 0.9 matches 3D-LEX elbow/reach on signs made in the air (depth_eval.py); contact signs are then snapped back
CONTACT_SNAP = os.environ.get("ASLC_CONTACT_SNAP", "1") == "1"   # contact_snap.snap for ASL-LEX body/head contact
SPINE_FOLD = os.environ.get("ASLC_SPINE_FOLD", "1") == "1"   # env too (0 = off), so consensus.py builds WLASL clips alike
HEAD_LEVEL = os.environ.get("ASLC_HEAD_LEVEL", "1") == "1"       # fold_spine head_pitch (head faces the viewer)
TORSO_ANCHOR = os.environ.get("ASLC_TORSO_ANCHOR", "0") == "1"   # contact_snap.anchor; off: video2d_check rejects it


def load_clip(stem, seg, mean_rh, mean_lh_model, feats=FEATS):
    with h5py.File(feats / "nlf" / f"{stem}.h5") as n, h5py.File(feats / "wilor" / f"{stem}.h5") as w:
        fps = float(n.attrs["fps"])
        nv = n["valid"][:, 0]
        body_aa = n["params/smplx/body_pose"][:, 0].reshape(-1, 21, 3)
        kp2 = n["keypoints_2d"][:, 0]                         # (T, 55, 2): 20 = left wrist, 21 = right wrist
        wv, wb, whp = w["valid"][:], w["boxes"][:], w["params/mano/hand_pose"][:]
        wk2 = w["keypoints_2d"][:]                            # (T, K, 21, 2); 0 = wrist
        go = n["params/smplx/global_orient"][:, 0]
        nk3 = n["keypoints_3d"][:, 0]
        wk3 = w["keypoints_3d"][:]
        whand = w["handedness"][:]
    T = min(len(nv), len(wv))

    body = Rotation.from_rotvec(body_aa[:T].reshape(-1, 3)).as_matrix().reshape(T, 21, 3, 3)
    if not nv[:T].any():
        return None, "no NLF person"
    hands = {"rh": np.zeros((T, 15, 3, 3)), "lh": np.zeros((T, 15, 3, 3))}
    ok = {"rh": np.zeros(T, bool), "lh": np.zeros(T, bool)}
    det = {"rh": [], "lh": []}
    for t in range(T):
        best = {}
        for k in np.flatnonzero(wv[t]):
            d = {s: np.linalg.norm(wk2[t, k, 0] - kp2[t, j]) for s, j in (("lh", 20), ("rh", 21))}
            s = min(d, key=d.get)
            if s not in best or d[s] < best[s][0]:            # per side: the detection nearest that wrist
                best[s] = (d[s], k)
        for s, (_, k) in best.items():
            hands[s][t], ok[s][t] = whp[t, k], True
            det[s].append((t, k))
    arm_dz = {}
    if ARM_DEPTH:
        body, arm_dz = fix_arm_depth(body, go[:T], nk3[:T], kp2[:T], nv[:T], det, wk3, wk2, whand)
    body = fill(body, nv[:T])
    v = nv[:T].astype(bool)                                   # the head's pitch towards the camera, as filmed
    Gh = FLIP @ Rotation.from_rotvec(go[:T][v]).as_matrix()
    for j in (3, 6, 9, 12, 15):
        Gh = Gh @ body[v, j - 1]
    head_cam_pitch = float(np.median(_pitch(Gh)))
    spikes = {}
    for side in ok:
        ok[side], spikes[side] = drop_spikes(hands[side], ok[side])
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
    obs = {"body": nv[a:b].astype(bool), "rh": ok["rh"][a:b], "lh": ok["lh"][a:b]}
    if mirrored:
        body = mirror_body(body)
        rh, lh = lh, rh                                       # model frame: mirroring = swapping the streams
        cov, spikes = {"rh": cov["lh"], "lh": cov["rh"]}, {"rh": spikes["lh"], "lh": spikes["rh"]}
        obs["rh"], obs["lh"] = obs["lh"], obs["rh"]
    sig = {}
    (body, sig["body"]), (rh, sig["rh"]), (lh, sig["lh"]) = (adaptive_smooth(x, obs[k]) for x, k in
                                                             ((body, "body"), (rh, "rh"), (lh, "lh")))
    body, rh, lh = (resample(x, fps) for x in (body, rh, lh))
    anchored = None
    if SPINE_FOLD:
        fb = fold_spine(body)
        head_fix = head_cam_pitch - float(np.median(_pitch(fb[:, 11] @ fb[:, 14]))) if HEAD_LEVEL else 0.0
        fb = fold_spine(body, head_pitch=head_fix)
        if TORSO_ANCHOR:                                       # hands on the torso go back where NLF put them
            import contact_snap as cs
            (b60, _, _), anchored = cs.anchor((r6(body[:, 11:21]), r6(rh), r6(lh)), (r6(fb[:, 11:21]), r6(rh), r6(lh)))
            fb[:, 11:21] = from6(b60.reshape(len(fb), -1))
        body = fb
    meta = {"fps_src": fps, "frames_src": int(T), "trim": [int(a), int(b)], "fast_signs": int((sg == 2).sum()),
            "hand_frames": cov, "mirrored": bool(mirrored), "fps": FPS,      # hand_frames: after mirroring (rh = dominant)
            "smooth_sigma_src_frames": sig, "spikes_dropped": spikes,
            "arm_depth_dz_m": arm_dz, "spine_fold": FOLD_ALPHA if SPINE_FOLD else 0.0,
            "torso_anchor_mean_weight": anchored, "head_cam_pitch_deg": head_cam_pitch,
            "head_fix_deg": head_fix if SPINE_FOLD else 0.0}                                 # before mirroring (camera sides)
    return {"body": body, "rh": rh, "lh_model": lh}, meta


_LEX = None


def apply_contact(s, meta, gloss):
    """ASL-LEX contact snap on a built clip (contact_snap.py): hands of signs coded as body/head contact touch."""
    global _LEX
    if not CONTACT_SNAP:
        return s
    import contact_snap as cs
    if _LEX is None:
        _LEX = cs.lex_codes()
    r = _LEX.get(gloss)
    if not cs.is_contact(r):
        return s
    out, info = cs.snap((r6(s["body"][:, 11:21]), r6(s["rh"]), r6(s["lh_model"])), r["MajorLocation.2.0"])
    meta["contact_snap"] = info
    if out is not None:
        s = dict(s)
        s["body"] = s["body"].copy()
        s["body"][:, 11:21] = from6(out[0].reshape(len(s["body"]), -1))
    return s


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
    if not getattr(wm, "_J_cpu", False):         # load the joint model once (full_merge calls this per gloss)
        wm._J, wm._J_cpu = wm.sm.SoKeJoints(flat_hand_mean=True, device="cpu"), True
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
        # nor a truncated clip: the metric resamples to 64 frames, so a 0.25 s fragment can look typical
        # (GOOD's consensus came out 6 frames long); durations below 0.6 x the gloss median are excluded
        dur = np.array([len(np.load(d / "body_rot6d.npy")) for d in dirs])
        long_ok = dur >= 0.6 * np.median(dur)
        e["short_clips"] = [c for c, ok in zip(e["clips"], long_ok) if not ok]
        # nor, for ASL-LEX contact signs, a clip whose hand never reaches the coded region (contact snap skipped)
        snap_ok = np.array(["skipped" not in json.loads((d / "meta.json").read_text()).get("contact_snap", {})
                            for d in dirs])
        e["no_contact_clips"] = [c for c, ok in zip(e["clips"], snap_ok) if not ok]
        cov = np.where(long_ok & snap_ok, cov, -1.0)
        good = cov >= 0.6 if (cov >= 0.6).sum() >= 3 else cov >= np.sort(cov)[-3]
        e["medoid"] = e["clips"][int(np.argmin(np.where(good, med, np.inf)))]
        e["spread_median"] = float(np.median(med))
        e["ranked"] = [e["clips"][i] for i in np.argsort(med)]


DEMO = ("HOW1 HOW2 YOU YOUR SCHOOL WHERE PLEASE HELP BATHROOM SORRY SEE TOMORROW WHO SLOW HELLO ME GOOD NEED NICE "
        "LEARN LANGUAGE SIGNLANGUAGE MEET THANKYOU NAME WHAT1 WHAT2").split()


def main():
    global OUT, ARM_DEPTH, SPINE_FOLD
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--glosses", nargs="*", default=DEMO, help="'all' for every gloss with features")
    ap.add_argument("--out", type=Path, default=None, help="default data/asl_citizen/dict")
    ap.add_argument("--no-arm-depth", action="store_true")
    ap.add_argument("--no-spine-fold", action="store_true")
    ap.add_argument("--shard", default=None, help="k/n: clips only (no medoids/index/LMDB), for the full build")
    ap.add_argument("--skip-done", action="store_true", help="with --shard: skip clips already saved")
    a = ap.parse_args()
    sk, sn = map(int, a.shard.split("/")) if a.shard else (0, 1)
    OUT, ARM_DEPTH, SPINE_FOLD = a.out or OUT, not a.no_arm_depth, SPINE_FOLD and not a.no_spine_fold
    keep = None if a.glosses == ["all"] else set(a.glosses)
    sys.path.insert(0, str(REPO / "dictionary"))
    fast = torch.load(FEATS / "fast.pt", weights_only=False)
    st = np.load(REPO / "outputs" / "static_How2Sign.npz")
    gloss_of = {}
    for s in ("train", "val", "test"):
        for r in csv.DictReader(open(AC / "ASL_Citizen" / "splits" / f"{s}.csv")):
            gloss_of[Path(r["Video file"]).stem] = (r["Gloss"], r["Participant ID"], s)
    index, records, fails = {}, [], {}
    for n, h in enumerate(sorted((FEATS / "nlf").glob("*.h5"))):
        stem = h.stem
        if n % sn != sk:
            continue
        if stem not in gloss_of or not (FEATS / "wilor" / h.name).exists():
            continue
        g, pid, split = gloss_of[stem]
        if keep is not None and g not in keep:
            continue
        if a.shard and a.skip_done and (OUT / "clips" / g / stem / "meta.json").exists():
            continue
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
        s = apply_contact(s, meta, g)
        save_streams(OUT / "clips" / g / stem, s, meta)
        index.setdefault(g, {"clips": []})["clips"].append(stem)
        if a.shard:
            if sum(len(e["clips"]) for e in index.values()) % 200 == 0:
                print(sum(len(e["clips"]) for e in index.values()), "clips", len(fails), "failed", flush=True)
            continue
        records.append((f"{g}/{stem}", lmdb_record(g, s)))
    if a.shard:
        (OUT / f"shard_{sk}of{sn}.json").write_text(json.dumps({"glosses": index, "failed": fails}))
        print(f"shard {sk}/{sn}: {sum(len(e['clips']) for e in index.values())} clips, {len(fails)} failed", flush=True)
        return
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
