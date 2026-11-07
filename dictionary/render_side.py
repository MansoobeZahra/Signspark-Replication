"""Side-view comparison sheet (elbow bend / hand depth are invisible from the front).

    ../ver-01/.venv-signspark/bin/python dictionary/render_side.py OUT.png GLOSS:DIR[:LABEL] ...
Each argument is one row: 6 frames evenly over the sign, camera at the signer's right, 90 degrees.
"""
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(REPO.parent / "ver_02"))
import cv2  # noqa: E402
import imageio  # noqa: E402
import render_signspark_reference as rsr  # noqa: E402


def row(d, label, n=6):
    V, J, faces = rsr.human_mesh(Path(d))
    R = rsr.Renderer(faces)
    tgt = J[0, 12] + np.array([0, -0.15, 0.1])
    tiles = []
    for t in np.linspace(0, len(V) - 1, n).round().astype(int):
        img = R.render(V[t], tgt + np.array([-1.6, 0.0, 0.0]), tgt)       # signer's right side
        img = cv2.resize(img, (240, 240))
        rsr.label(img, [label if t == 0 else "", f"f{t}"])
        tiles.append(img)
    R.r.delete()
    return np.concatenate(tiles, 1)


if __name__ == "__main__":
    rows = []
    for a in sys.argv[2:]:
        g, d, *lab = a.split(":")
        rows.append(row(d, lab[0] if lab else g))
    imageio.imwrite(sys.argv[1], np.concatenate(rows, 0))
