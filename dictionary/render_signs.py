"""Render every dictionary sign on its own for visual review: video.mp4 + sheet.png (8 frames evenly across the sign,
full body + dominant-hand close-up, ver_02 renderer, EGL).

    ../ver-01/.venv-signspark/bin/python dictionary/render_signs.py [--src final|consensus] [GLOSS ...]
Output: outputs/dictionary/signs/<src>/<GLOSS>/{video.mp4, sheet.png}
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
DICT = Path(os.environ.get("ASLC_DICT", REPO / "data" / "asl_citizen" / "dict"))
OUT = REPO / "outputs" / "dictionary" / "signs" / DICT.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glosses", nargs="*")
    ap.add_argument("--src", default="final")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    sys.path.insert(0, str(ROOT / "ver_02"))
    import cv2
    import imageio
    import render_signspark_reference as rsr
    dirs = [DICT / a.src / g for g in a.glosses] if a.glosses else sorted(p.parent for p in (DICT / a.src).glob("*/meta.json"))
    for d in dirs:
        meta = json.loads((d / "meta.json").read_text())
        V, J, faces = rsr.human_mesh(d)
        R = rsr.Renderer(faces)
        label = f"{d.name}  [{meta.get('source', a.src)[:40]}]"
        frames = [rsr.frame_pair(R, V, J, t, 24, label) for t in range(len(V))]
        R.r.delete()
        o = OUT / a.src / d.name
        o.mkdir(parents=True, exist_ok=True)
        w = imageio.get_writer(o / "video.mp4", fps=24, codec="libx264", pixelformat="yuv420p", quality=8)
        for f in frames:
            w.append_data(f)
        w.close()
        picks = np.linspace(0, len(frames) - 1, a.n).round().astype(int)
        tiles = [cv2.resize(frames[t], (480, 240)) for t in picks]
        imageio.imwrite(o / "sheet.png", np.concatenate([np.concatenate(tiles[i:i + 2], 1) for i in range(0, len(tiles), 2)], 0))
        print(d.name, len(frames), "frames", flush=True)


if __name__ == "__main__":
    main()
