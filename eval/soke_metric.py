"""SOKE's DTW joint-position metric (the one SignSparK's paper reports), ported for SignSparK outputs.

Source: github.com/2000ZRL/SOKE @ 5cbc55d (2026-02-21), mGPT/metrics/t2m.py (TM2TMetrics.update),
mGPT/metrics/dtw.py, mGPT/utils/human_models.py, mGPT/data/H2S.py (feats2joints). Definition, as coded there:

  * SMPL-X neutral, use_face_contour=True (144 joints), root/legs/spine zero, betas = H2S.py's fixed
    shape_param, eye poses zero. Hand pose convention: SignSparK's (flat_hand_mean=True, its visualize.py);
    SOKE's own data uses the smplx default (False) -- a parameter here.
  * body  = joints Neck, L/R Shoulder, L/R Elbow, L/R Wrist, L/R Ear, L/R Eye, Nose (12 joints, taken from the
            full 144-joint output); hands = 21 joints each (wrist, 15 regressed, 5 tip vertices).
  * per-frame distance l2_dist_align(pred, gt): align pred to gt, then mean joint L2 over the part:
      DTW-JPE    align_idx=0:    translate so joint 0 matches (body: pelvis of the full set; hand: wrist)
      DTW-PA-JPE align_idx=None: similarity Procrustes (rotation, scale, translation) over ALL joints passed
                                 (body: all 144; hand: its 21), then distance on the part's joints
    (SOKE's released code computes only the align_idx=0 variant but names it "PA"; its comment says
     align_idx=None is DTW-PA-JPE. Both are computed here, each named for what it is.)
  * dtw(pred, gt): standard DTW (warp 1, no window, s=1); the score is the ACCUMULATED cost D[-1,-1] -- a SUM
    of per-frame mean joint errors in metres along the path, not divided by path length or frame count.
    compute() averages it over sequences. Units: metres x path steps, as the paper's table.

`dtw_fast` / `pair_costs` reproduce SOKE's dtw + l2_dist_align exactly (checked against the verbatim
reference copies below by `python soke_metric.py`), but vectorised: the DP runs along anti-diagonals and
the n x m Procrustes fits are batched on the GPU.
"""

from functools import partial
from math import isinf
from pathlib import Path

import numpy as np
import smplx
import torch
from numpy import argmin, array, full, inf, zeros

ROOT = Path(__file__).resolve().parents[1]   # repo root; SMPL-X model at ROOT/smplx/SMPLX_NEUTRAL.npz
SOKE_BETAS = [-0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172,
              0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842]   # H2S.py feats2joints
# human_models.py: joint_idx[joint_part['upper_body']] = Neck..R_Wrist + L_Ear..Nose, in raw SMPL-X indices
BODY_IDX = [12, 16, 17, 18, 19, 20, 21, 59, 58, 57, 56, 55]
PARTS = ("body", "lhand", "rhand")


# ============================== verbatim SOKE reference (dtw.py, human_models.py) ==============================
def dtw(x, y, dist, warp=1, w=inf, s=1.0):
    assert len(x)
    assert len(y)
    assert isinf(w) or (w >= abs(len(x) - len(y)))
    assert s > 0
    r, c = len(x), len(y)
    if not isinf(w):
        D0 = full((r + 1, c + 1), inf)
        for i in range(1, r + 1):
            D0[i, max(1, i - w):min(c + 1, i + w + 1)] = 0
        D0[0, 0] = 0
    else:
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf
        D0[1:, 0] = inf
    D1 = D0[1:, 1:]  # view
    for i in range(r):
        for j in range(c):
            if (isinf(w) or (max(0, i - w) <= j <= min(c, i + w))):
                D1[i, j] = dist(x[i], y[j])
    C = D1.copy()
    jrange = range(c)
    for i in range(r):
        if not isinf(w):
            jrange = range(max(0, i - w), min(c, i + w + 1))
        for j in jrange:
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                i_k = min(i + k, r)
                j_k = min(j + k, c)
                min_list += [D0[i_k, j] * s, D0[i, j_k] * s]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1], C, D1, path


def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)


def l2_dist_align(x, y, wanted=None, align_idx=None):
    if align_idx is None:
        x = rigid_align(x, y)
    else:
        x = x - x[align_idx:align_idx+1] + y[align_idx:align_idx+1]
    if wanted is not None:
        x = x[wanted]
        y = y[wanted]
    dist = np.mean(np.sqrt(((x-y)**2).sum(axis=1)))
    return dist


def rigid_transform_3D(A, B):
    n, dim = A.shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))

    varP = np.var(A, axis=0).sum()
    c = 1 / varP * np.sum(s)

    t = -np.dot(c * R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return c, R, t


def rigid_align(A, B):
    c, R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(c * R, np.transpose(A))) + t
    return A2


# ============================== fast, equivalent implementation ==============================
def pair_costs(x, y, wanted=None, align_idx=None, device="cuda"):
    """C[i, j] = l2_dist_align(x[i], y[j], wanted, align_idx) for all pairs. x (n, J, 3), y (m, J, 3)."""
    X = torch.as_tensor(x, dtype=torch.float64, device=device)
    Y = torch.as_tensor(y, dtype=torch.float64, device=device)
    w = torch.arange(X.shape[1], device=device) if wanted is None else torch.as_tensor(wanted, device=device)
    if align_idx is not None:
        xr = X[:, w] - X[:, align_idx:align_idx + 1]                  # (n, k, 3)
        yr = Y[:, w] - Y[:, align_idx:align_idx + 1]                  # (m, k, 3)
        return (xr[:, None] - yr[None]).norm(dim=-1).mean(-1).cpu().numpy()
    n, J, _ = X.shape
    cA, cB = X.mean(1), Y.mean(1)                                     # (n, 3), (m, 3)
    Ac, Bc = X - cA[:, None], Y - cB[:, None]
    varP = Ac.pow(2).sum((1, 2)) / J                                  # np.var(A, axis=0).sum()
    out = np.empty((n, Y.shape[0]))
    for i0 in range(0, n, 64):                                        # chunk rows to bound memory
        i1 = min(n, i0 + 64)
        H = torch.einsum("ika,jkb->ijab", Ac[i0:i1], Bc) / J          # (r, m, 3, 3)
        U, s, Vh = torch.linalg.svd(H)
        R = Vh.transpose(-1, -2) @ U.transpose(-1, -2)
        neg = torch.linalg.det(R) < 0
        s = torch.where(neg[..., None], torch.cat([s[..., :2], -s[..., 2:]], -1), s)
        Vh = torch.where(neg[..., None, None], torch.cat([Vh[..., :2, :], -Vh[..., 2:, :]], -2), Vh)
        R = Vh.transpose(-1, -2) @ U.transpose(-1, -2)
        c = s.sum(-1) / varP[i0:i1, None]                             # (r, m)
        # aligned A_k = c R (A_k - cA) + cB, on the wanted joints only
        Aw = torch.einsum("ijab,ikb->ijka", R, Ac[i0:i1][:, w]) * c[..., None, None] + cB[None, :, None]
        out[i0:i1] = (Aw - Y[None, :, w]).norm(dim=-1).mean(-1).cpu().numpy()
    return out


def dtw_fast(C):
    """SOKE dtw() accumulated cost D1[-1, -1] (warp 1, w inf, s 1) for a precomputed cost matrix C (r, c)."""
    r, c = C.shape
    D = np.full((r + 1, c + 1), np.inf)
    D[0, 0] = 0.0
    for d in range(r + c - 1):                                       # anti-diagonal i + j = d
        i = np.arange(max(0, d - c + 1), min(r, d + 1))
        j = d - i
        D[i + 1, j + 1] = C[i, j] + np.minimum(np.minimum(D[i, j], D[i, j + 1]), D[i + 1, j])
    return float(D[r, c])


# ============================== SMPL-X joints as SOKE computes them ==============================
class SoKeJoints:
    def __init__(self, flat_hand_mean=True, device="cuda"):
        self.device = device
        self.layer = smplx.create(str(ROOT), "smplx", gender="neutral", ext="npz", use_pca=False,
                                  use_face_contour=True, flat_hand_mean=flat_hand_mean).to(device)
        reg = self.layer.J_regressor.cpu().numpy()
        eye = np.eye(reg.shape[1])
        # human_models.py make_hand_regressor (verbatim indices)
        self.hand_reg = {
            "lhand": np.concatenate((reg[[20, 37, 38, 39], :], eye[5361, None], reg[[25, 26, 27], :], eye[4933, None],
                                     reg[[28, 29, 30], :], eye[5058, None], reg[[34, 35, 36], :], eye[5169, None],
                                     reg[[31, 32, 33], :], eye[5286, None])),
            "rhand": np.concatenate((reg[[21, 52, 53, 54], :], eye[8079, None], reg[[40, 41, 42], :], eye[7669, None],
                                     reg[[43, 44, 45], :], eye[7794, None], reg[[49, 50, 51], :], eye[7905, None],
                                     reg[[46, 47, 48], :], eye[8022, None]))}
        self.hand_reg = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in self.hand_reg.items()}

    @torch.no_grad()
    def __call__(self, body_aa, lhand_aa, rhand_aa):
        """(T, 21, 3) body_pose (legs/spine already zero), (T, 15, 3) hands, SMPL-X convention ->
        {'body': (T, 144, 3), 'lhand': (T, 21, 3), 'rhand': (T, 21, 3)} in metres."""
        T = len(body_aa)
        f = lambda a: torch.as_tensor(np.asarray(a).reshape(T, -1), dtype=torch.float32, device=self.device)  # noqa
        z = torch.zeros(T, 3, device=self.device)
        out = self.layer(betas=torch.tensor([SOKE_BETAS], device=self.device).repeat(T, 1),
                         global_orient=z, body_pose=f(body_aa), left_hand_pose=f(lhand_aa),
                         right_hand_pose=f(rhand_aa), jaw_pose=z, leye_pose=z, reye_pose=z,
                         expression=torch.zeros(T, 10, device=self.device))
        V = out.vertices
        return {"body": out.joints.cpu().numpy().astype(np.float64),
                "lhand": torch.einsum("kv,tvc->tkc", self.hand_reg["lhand"], V).cpu().numpy().astype(np.float64),
                "rhand": torch.einsum("kv,tvc->tkc", self.hand_reg["rhand"], V).cpu().numpy().astype(np.float64)}


def score(pred, gt, device="cuda"):
    """pred/gt: SoKeJoints outputs. Returns {'<part>_jpe', '<part>_pa_jpe'} (SOKE units: summed metres)."""
    res = {}
    for part in PARTS:
        wanted = BODY_IDX if part == "body" else None
        res[f"{part}_jpe"] = dtw_fast(pair_costs(pred[part], gt[part], wanted, 0, device))
        res[f"{part}_pa_jpe"] = dtw_fast(pair_costs(pred[part], gt[part], wanted, None, device))
        res[f"{part}_steps"] = max(len(pred[part]), len(gt[part]))
    return res


def _selftest():
    rng = np.random.default_rng(0)
    for J, wanted, n, m in ((144, BODY_IDX, 23, 17), (21, None, 19, 26)):
        x = rng.normal(size=(n, J, 3)) * 0.2
        y = x[np.linspace(0, n - 1, m).astype(int)] + rng.normal(size=(m, J, 3)) * 0.03
        y = y @ np.linalg.qr(rng.normal(size=(3, 3)))[0].T                # include a reflection sometimes
        for align in (0, None):
            ref = dtw(x, y, partial(l2_dist_align, wanted=wanted, align_idx=align))[0]
            fast = dtw_fast(pair_costs(x, y, wanted, align))
            print(f"J={J} align={align}: reference {ref:.10f}  fast {fast:.10f}  |diff| {abs(ref - fast):.2e}")
            assert abs(ref - fast) < 1e-9 * max(1, ref)


if __name__ == "__main__":
    _selftest()
