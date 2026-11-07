"""2D-consistency check: does the rendered avatar (front view) put the wrist where the signer's wrist is in the video?

Height of the dominant wrist above the shoulder line, in shoulder widths, over the raised half of the sign:
video = NLF 2D keypoints (its fit to the image, the closest thing to the film itself), render = the stream's
SMPL-X joints seen from the front (x/y). Variants: unfolded (arms relative to NLF's bent spine3), fold (spine bend
folded into neck/collars, aslc_build default) and fold fractions. Signed difference render - video per clip.

    CUDA_VISIBLE_DEVICES= .venv-sltk/bin/python dictionary/video2d_check.py [--glosses all|G ...] [--per 4]
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aslc_build as ab  # noqa: E402
from compare_proto import wm  # noqa: E402

ALPHAS = (0.0, 0.5, 0.75, 1.0, 1.25)


def render_h(body, rh, lh):
    J = wm.stream_joints(ab.r6(body[:, 11:21]), ab.r6(rh), ab.r6(lh))
    B = J["body"]
    sh, sw = (B[:, 16] + B[:, 17]) / 2, np.linalg.norm(B[:, 16, :2] - B[:, 17, :2], axis=1)
    return (B[:, 21, 1] - sh[:, 1]) / sw


def video_h(stem, meta):
    a, b = meta["trim"]
    with h5py.File(ab.FEATS / "nlf" / f"{stem}.h5") as n:
        k, v = n["keypoints_2d"][a:b, 0], n["valid"][a:b, 0]
    k = k[v.astype(bool)]
    w = 20 if meta["mirrored"] else 21
    sh, sw = (k[:, 16] + k[:, 17]) / 2, np.linalg.norm(k[:, 16] - k[:, 17], axis=1)
    return (sh[:, 1] - k[:, w, 1]) / sw


def top(v):
    return float(np.median(v[v >= np.quantile(v, 0.5)])) if len(v) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glosses", nargs="*", default=None)
    ap.add_argument("--per", type=int, default=4)
    a = ap.parse_args()
    idx = json.loads((ab.OUT / "index.json").read_text())["glosses"]
    gl = sorted(idx) if not a.glosses else a.glosses
    fast = torch.load(ab.FEATS / "fast.pt", weights_only=False)
    st = np.load(ab.REPO / "outputs" / "static_How2Sign.npz")
    ab.SPINE_FOLD = False
    rows = []
    for g in gl:
        for stem in idx[g].get("ranked", idx[g]["clips"])[: a.per]:
            sg = fast.get(stem)
            s, meta = ab.load_clip(stem, sg.numpy() if sg is not None else None, st["rh"], st["lh_rframe"])
            if s is None:
                continue
            row = {"gloss": g, "stem": stem, "video": top(video_h(stem, meta))}
            for al in ALPHAS:
                row[al] = top(render_h(ab.fold_spine(s["body"], al), s["rh"], s["lh_model"]))
            rows.append(row)
        sel = [r for r in rows if r["gloss"] == g]
        print(f"{g:13} video {np.nanmedian([r['video'] for r in sel]):+.2f} | " +
              " ".join(f"a{al}: {np.nanmedian([r[al] - r['video'] for r in sel]):+.2f}" for al in ALPHAS), flush=True)
    print("\nALL clips, median render - video (shoulder widths): " +
          " ".join(f"a{al}: {np.nanmedian([r[al] - r['video'] for r in rows]):+.3f} (|.| {np.nanmedian([abs(r[al] - r['video']) for r in rows]):.3f})"
                   for al in ALPHAS))
    (ab.REPO / "outputs" / "dictionary" / "video2d_check.json").write_text(json.dumps(rows, default=str))


if __name__ == "__main__":
    main()
