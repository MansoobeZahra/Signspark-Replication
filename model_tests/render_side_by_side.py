"""Side-by-side renders: real ground truth vs model output, BOTH through the same conversion code and the same
renderer (ver_02/render_signspark_reference.py's Renderer + frame_pair, imported unchanged).

SignSparK: rows = GT | text-only (official sample.py, keyframes_mask_all) | keyframe mode (official sample.py).
  All three: SignSparK tools/visualize.py:build_smplx_input -> SMPL-X (flat_hand_mean=True), left-hand flip per
  the measured convention of each array (score_signspark.left_convention).
MaDiS: rows = GT (feats_ref) | MaDiS (feats_rst), both from the official test.py dump: de-normalised with MaDiS's
  mean/std -> ../ver_03_madis/convert_madis.convert -> the same build_smplx_input + SMPL-X as SignSparK.
Shorter sequences hold their last frame. Output per clip: sheet.png (6 time points) and video.mp4.

    .venv-signspark/bin/python render_side_by_side.py signspark --t2p <hand.npy> <body.npy> --kf2p <hand.npy> <body.npy>
    .venv-signspark/bin/python render_side_by_side.py madis --pred-dir <dir with test_rank_0/*.pkl>
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import cv2
import imageio
import numpy as np
import smplx
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "ver_02"))
sys.path.insert(1, str(ROOT.parent / "ver_03_madis"))
import render_signspark_reference as rsr  # noqa: E402  (also puts SignSparK/tools on the path)
from score_signspark import load, left_convention  # noqa: E402
ssv = rsr.ssv
CLIPS = ["G1GUMky8kWc_20-2-rgb_front", "G095RWKQ39g_13-1-rgb_front", "G3FhmHz_7hs_6-10-rgb_front"]


def mesh(body, lh, rh, flip_left):
    t = lambda x: torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32)  # noqa: E731
    m = smplx.create(str(ROOT.parent / "ver_02"), model_type="smplx", gender="neutral", use_pca=False,
                     flat_hand_mean=True, num_betas=10, num_expression_coeffs=50, batch_size=len(body)).eval()
    kw = ssv.build_smplx_input(t(body), t(lh), t(rh), face=None, num_expression=50, flip_left_hand=flip_left)
    with torch.no_grad():
        o = m(**kw)
    V, J = o.vertices.numpy(), o.joints.numpy()
    return V - J[:, :1], J - J[:, :1], m.faces


class SoftRenderer:
    """CPU fallback when EGL is unavailable (here: NVIDIA kernel 595.91 vs userspace 595.99 after an unrebooted
    driver update; CUDA still works, EGL does not). Orthographic painter's-algorithm mesh render with Lambert shading
    (matplotlib). Same meshes, same frame layout as ver_02's frame_pair: front view | dominant (right) hand close-up."""

    def __init__(self, faces):
        self.faces = np.asarray(faces)

    def view(self, V, center, half, size):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        tri = V[self.faces]                                           # (F, 3, 3)
        keep = np.all(np.abs(tri[..., :2] - center[:2]).max(-1) < half * 1.3, axis=1)
        tri = tri[keep]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        shade = np.clip(n @ np.array([0.3, 0.4, 0.87]), 0, 1) * 0.75 + 0.2
        order = np.argsort(tri[:, :, 2].mean(1))                      # far (small z) first; camera at +z
        fig = plt.figure(figsize=(size / 100, size / 100), dpi=100, facecolor=(0.12, 0.12, 0.14))
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor((0.12, 0.12, 0.14))
        col = np.array([0.72, 0.52, 0.42])[None] * shade[order, None]
        ax.add_collection(PolyCollection(tri[order, :, :2], facecolors=col, edgecolors="none"))
        ax.set_xlim(center[0] - half, center[0] + half); ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_aspect("equal"); ax.axis("off")
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return img

    def frame_pair(self, V, J, t, fps, text):
        front = self.view(V[t], np.array([0.0, 0.25, 0.0]), 0.75, rsr.SIZE)
        c = 0.5 * (J[t, 21] + J[t, 45])
        close = self.view(V[t], c, 0.13, rsr.SIZE)
        rsr.label(front, [text[:60], f"frame {t:3d}  t = {t / fps:5.2f} s @ {fps:g} fps"])
        rsr.label(close, ["dominant (signer's right) hand close-up"])
        return np.concatenate([front, close], axis=1)


def render(out, text, panels, fps):
    """panels: list of (label, V, J). One frame = panels stacked vertically, each front | right-hand close-up."""
    out.mkdir(parents=True, exist_ok=True)
    try:
        R = rsr.Renderer(FACES[0])
        pair = lambda V, J, k: rsr.frame_pair(R, V, J, k, fps, text)  # noqa: E731
    except Exception as e:                                             # EGL unavailable -> CPU fallback
        print(f"EGL renderer unavailable ({type(e).__name__}); using the CPU fallback renderer", flush=True)
        R = None
        soft = SoftRenderer(FACES[0])
        pair = lambda V, J, k: soft.frame_pair(V, J, k, fps, text)    # noqa: E731
    T = max(len(V) for _, V, _ in panels)

    def frame(t):
        rows = []
        for lab, V, J in panels:
            k = min(t, len(V) - 1)
            img = pair(V, J, k)
            bar = np.full((34, img.shape[1], 3), 45, np.uint8)
            rsr.header(bar, [f"{lab}   ({len(V)} frames{', holding last' if t >= len(V) else ''})"])
            rows.append(np.concatenate([bar, img], 0))
        return np.concatenate(rows, 0)

    picks = np.linspace(0, T - 1, 6).astype(int)
    tiles = [cv2.resize(frame(t), (480, int(round(274 * len(panels) / 1.0)))) for t in picks]
    imageio.imwrite(out / "sheet.png", np.concatenate(tiles, 1))
    step = 1 if R is not None else VIDEO_STEP                          # CPU fallback: every VIDEO_STEP-th frame
    w = imageio.get_writer(out / "video.mp4", fps=fps / step, codec="libx264", pixelformat="yuv420p", quality=8)
    for t in range(0, T, step):
        w.append_data(frame(t))
    w.close()
    if R is not None:
        R.r.delete()
    print(f"wrote {out.relative_to(ROOT)}/ ({T} frames)", flush=True)


FACES = [None]
VIDEO_STEP = 3


def signspark(t2p, kf2p):
    out = ROOT / "renders" / "signspark"
    D = {"t2p": (load(t2p[0]), load(t2p[1])), "kf2p": (load(kf2p[0]), load(kf2p[1]))}
    for clip in CLIPS:
        panels, text = [], None
        for mode in ("gt", "t2p", "kf2p"):
            H, B = D["t2p" if mode == "gt" else mode]
            hn, bn = list(H["video_names"]), list(B["video_names"])
            k, b = hn.index(clip), bn.index(clip)          # k = left item, k+1 = right item
            L = int(H["lengths"][k])
            src = "gt_poses" if mode == "gt" else "pred_poses"
            lh, rh = H[src][k].T[:L], H[src][k + 1].T[:L]
            body = B[src][b].T[:L]
            conv, _ = left_convention(lh, rh)
            V, J, faces = mesh(body, lh, rh, conv == "right_frame")
            FACES[0] = faces
            text = str(H["text"][k])
            lab = {"gt": "GROUND TRUTH (LMDB)", "t2p": "SignSparK text-only (official)", "kf2p": "SignSparK keyframe mode (official)"}[mode]
            panels.append((f"{lab} | left: {conv}", V, J))
        render(out / clip, text, panels, fps=20)


def madis(pred_dir):
    from convert_madis import convert
    out = ROOT / "renders" / "madis"
    mean = torch.load(ROOT / "data" / "CSL-Daily" / "mean.pt", map_location="cpu", weights_only=True)
    std = torch.load(ROOT / "data" / "CSL-Daily" / "std.pt", map_location="cpu", weights_only=True)
    # H2SDataModule: drop the 36 lower-body dims, then drop shape while keeping expression
    mean, std = mean[36:], std[36:]
    mean = torch.cat([mean[:-20], mean[-10:]]).numpy()
    std = torch.cat([std[:-20], std[-10:]]).numpy()
    for clip in CLIPS:
        d = pickle.load(open(Path(pred_dir) / f"{clip}.pkl", "rb"))
        panels = []
        for key, lab in (("feats_ref", "GROUND TRUTH (SOKE fittings)"), ("feats_rst", "MaDiS (official test.py)")):
            body, rh, lh = convert(d[key] * std + mean)
            V, J, faces = mesh(body, lh, rh, True)      # convert() puts the left hand in the right-hand frame
            FACES[0] = faces
            panels.append((lab, V, J))
        render(out / clip, d["text"], panels, fps=24)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["signspark", "madis"])
    ap.add_argument("--t2p", nargs=2)
    ap.add_argument("--kf2p", nargs=2)
    ap.add_argument("--pred-dir")
    a = ap.parse_args()
    signspark(a.t2p, a.kf2p) if a.model == "signspark" else madis(a.pred_dir)
