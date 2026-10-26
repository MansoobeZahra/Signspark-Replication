"""Contact sheets for the prototype dictionary: WLASL video | its NLF SMPL-X fit | the 3D-LEX sign (if animated).

Uses ver_02's render_signspark_reference.py unchanged (SignSparK SMPL-X assembly, same camera; EGL, ~0.2 GB GPU).
Frames: 5 evenly spaced over the signing part of each sequence (rest phases trimmed as in word_metric, frac 0.35).

    ../ver-01/.venv-signspark/bin/python dictionary/render_proto.py [--glosses how you ...]
Output: outputs/dictionary/renders/<gloss>.png (one per clip row), index.png (first clip of every gloss)
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import cv2  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
sys.path.insert(0, str(ROOT / "ver_02"))
import render_signspark_reference as rsr  # noqa: E402

W = REPO / "data" / "wlasl"
LEX = ROOT / "ver_04_finetune" / "data" / "3dlex"
OUT = REPO / "outputs" / "dictionary" / "renders"
N = 5
TILE = 240


def active(J, frac=0.35):
    """[a, b) of the frames whose higher wrist is above frac of its range (word_metric_variants.trim)."""
    h = (J[:, [20, 21], 1] - J[:, :1, 1]).max(1)
    idx = np.where(h >= h.min() + frac * (h.max() - h.min()))[0]
    return (idx[0], idx[-1] + 1) if len(idx) and idx[-1] - idx[0] >= 4 else (0, len(J))


def picks(a, b):
    return np.linspace(a, b - 1, N).astype(int)


def render_row(d, title):
    V, J, faces = rsr.human_mesh(d)
    R = rsr.Renderer(faces)
    a, b = active(J)
    tiles = [cv2.resize(R.render(V[t], eye=[0.0, 0.25, 1.9], target=[0.0, 0.2, 0.0]), (TILE, TILE)) for t in picks(a, b)]
    R.r.delete()
    row = np.concatenate(tiles, 1)
    rsr.label(row, [title])
    return row, (a, b), len(V)


def video_row(p, span, n_fit, title):
    cap = cv2.VideoCapture(str(p))
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f[..., ::-1])
    a, b = (np.array(span) * len(fr) / n_fit).astype(int)
    tiles = []
    for t in picks(a, max(b, a + 1)):
        f = fr[min(t, len(fr) - 1)]
        s = TILE / max(f.shape[:2])
        f = cv2.resize(f, None, fx=s, fy=s)
        c = np.zeros((TILE, TILE, 3), np.uint8)
        y, x = (TILE - f.shape[0]) // 2, (TILE - f.shape[1]) // 2
        c[y:y + f.shape[0], x:x + f.shape[1]] = f
        tiles.append(c)
    row = np.concatenate(tiles, 1)
    rsr.label(row, [title])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glosses", nargs="*", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    gl = a.glosses or sorted(p.name for p in (W / "streams").iterdir())
    first = []
    for g in gl:
        rows = []
        lex = LEX / g.replace("_", " ")
        lex_row = render_row(lex, f"3D-LEX '{g}' (mocap)")[0] if (lex / "body_rot6d.npy").exists() else None
        for k, d in enumerate(sorted((W / "streams" / g).iterdir())):
            r, span, n = render_row(d, f"WLASL '{g}' {d.name}: NLF SMPL-X fit")
            vrow = video_row(W / "videos" / g / f"{d.name}.mp4", span, n, f"WLASL '{g}' {d.name}: video")
            rows += [vrow, r]
            if k == 0:
                first += [vrow, r] + ([lex_row] if lex_row is not None else [])
        if lex_row is not None:
            rows.append(lex_row)
        imageio.imwrite(OUT / f"{g}.png", np.concatenate(rows, 0))
        print(g, len(rows), "rows", flush=True)
    for i in range(0, len(first), 24):  # index in parts that stay viewable
        imageio.imwrite(OUT / f"index_{i // 24}.png", cv2.resize(np.concatenate(first[i:i + 24], 0), None, fx=0.6, fy=0.6))


if __name__ == "__main__":
    main()
