"""Where does the ASL Citizen hand-height bias come from? Matched-gloss comparison against 3D-LEX mocap.

For every gloss in both ASL Citizen (features extracted) and 3D-LEX, up to N ASL Citizen clips are built with
aslc_build.load_clip (arm-depth fix on) and compared with the 3D-LEX sign of the same gloss, on the same frames
definition (word_metric_variants.trim 0.35: the span where the hands are raised):
  * dominant wrist relative to the dominant shoulder (x lateral, y up, z forward), median over the span;
  * elbow angle (deg) over the span;
  * NLF torso lean: spine1-3 rotation relative to the pelvis (pitch, deg, + = forward) and the pelvis pitch in the
    camera (+ = top of the pelvis tilted toward the camera); SignSparK streams drop joints 1-11, so whatever lean NLF
    puts in the spine is rendered upright.
Variants of the ASL Citizen stream: "cur" (as built), "fold" (spine1-3 rotation folded into neck and collars, so
the arms keep their direction relative to the pelvis instead of relative to spine3) and "cam" (only the forward
pitch of spine3 relative to the camera's vertical folded in, one constant per clip).
Output: outputs/dictionary/height_probe.json (+ printed summary, overall and per ASL-LEX major location).

    CUDA_VISIBLE_DEVICES= .venv-sltk/bin/python dictionary/height_probe.py [--per 3] [--max 200]
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aslc_build as ab  # noqa: E402
from compare_proto import wm, wv  # noqa: E402

REPO = ab.REPO
OUTF = REPO / "outputs" / "dictionary" / "height_probe.json"


def norm(g):
    return re.sub("[^a-z]", "", g.lower())


def measure(body60, rh, lh):
    J = wv.trim(wm.stream_joints(body60, rh, lh), 0.35)
    B, R = J["body"], J["rhand"]
    rel = np.median(R[:, 0] - B[:, 17], 0)
    fo, up = R[:, 0] - B[:, 19], B[:, 17] - B[:, 19]
    el = np.degrees(np.arccos(((fo * up).sum(1) / np.linalg.norm(fo, axis=1) / np.linalg.norm(up, axis=1)).clip(-1, 1)))
    return {"x": float(rel[0]), "y": float(rel[1]), "z": float(rel[2]), "elbow": float(np.median(el)),
            "shoulder_y": float(np.median(B[:, 17, 1] - B[:, 12, 1]))}           # shoulder above/below neck


def pitch(R):
    """Forward pitch (deg) of a rotation's y axis in its parent frame (y up, z forward)."""
    y = R[..., :, 1]
    return np.degrees(np.arctan2(y[..., 2], y[..., 1]))


def fold(body, S=None):
    """Fold a torso rotation into neck + collars (joints 12, 13, 14): by default spine1-3 local rotations."""
    S = body[:, 2] @ body[:, 5] @ body[:, 8] if S is None else S
    b = body.copy()
    for j in (12, 13, 14):
        b[:, j - 1] = S @ body[:, j - 1]
    return b


def cam_pitch(stem):
    """Median forward pitch (deg) of the NLF pelvis and of spine3 relative to the camera's vertical, in the viewer
    frame (camera y down / z away -> y up / z toward the camera)."""
    with h5py.File(ab.FEATS / "nlf" / f"{stem}.h5") as n:
        v = n["valid"][:, 0]
        go = n["params/smplx/global_orient"][:, 0][v]
        bp = n["params/smplx/body_pose"][:, 0][v].reshape(-1, 21, 3)
    D = np.diag([1.0, -1.0, -1.0])
    G = D @ Rotation.from_rotvec(go).as_matrix()
    loc = Rotation.from_rotvec(bp.reshape(-1, 3)).as_matrix().reshape(-1, 21, 3, 3)
    G3 = G @ loc[:, 2] @ loc[:, 5] @ loc[:, 8]
    return float(np.median(pitch(G))), float(np.median(pitch(G3)))


def rx(deg):
    return Rotation.from_euler("x", deg, degrees=True).as_matrix()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=3)
    ap.add_argument("--max", type=int, default=200)
    a = ap.parse_args()
    lex = {norm(p.name): p for p in wm.LEX.iterdir() if (p / "meta.json").exists()}
    loc = {}
    for r in csv.DictReader(open(REPO / "data" / "asl_lex" / "signdata.csv", encoding="latin-1")):
        loc[r["Code"]] = r.get("MajorLocation.2.0", "")
    clips = {}
    for s in ("train", "val", "test"):
        for r in csv.DictReader(open(ab.AC / "ASL_Citizen" / "splits" / f"{s}.csv")):
            stem = Path(r["Video file"]).stem
            if norm(r["Gloss"]) in lex and (ab.FEATS / "nlf" / f"{stem}.h5").exists() and \
                    (ab.FEATS / "wilor" / f"{stem}.h5").exists():
                clips.setdefault(r["Gloss"], {"stems": [], "loc": loc.get(r["ASL-LEX Code"], "")})["stems"].append(stem)
    glosses = sorted(clips)[:: max(1, len(clips) // a.max)][: a.max]
    print(f"{len(clips)} matched glosses, probing {len(glosses)}", flush=True)
    fast = torch.load(ab.FEATS / "fast.pt", weights_only=False)
    st = np.load(REPO / "outputs" / "static_How2Sign.npz")
    rows = []
    for g in glosses:
        L = measure(*wm.load_streams(lex[norm(g)]))
        for stem in clips[g]["stems"][: a.per]:
            sg = fast.get(stem)
            try:
                s, meta = ab.load_clip(stem, sg.numpy() if sg is not None else None, st["rh"], st["lh_rframe"])
            except Exception as e:
                print(g, stem, type(e).__name__, e, flush=True)
                continue
            if s is None:
                continue
            b = s["body"]
            sp = Rotation.from_matrix((b[:, 2] @ b[:, 5] @ b[:, 8])).as_matrix()
            pp, p3 = cam_pitch(stem)
            row = {"gloss": g, "stem": stem, "loc": clips[g]["loc"], "lex": L, "pelvis_cam_pitch": pp,
                   "spine3_cam_pitch": p3,
                   "cam": measure(ab.r6(fold(b, rx(p3))[:, 11:21]), ab.r6(s["rh"]), ab.r6(s["lh_model"])),
                   "cur": measure(ab.r6(b[:, 11:21]), ab.r6(s["rh"]), ab.r6(s["lh_model"])),
                   "fold": measure(ab.r6(fold(b)[:, 11:21]), ab.r6(s["rh"]), ab.r6(s["lh_model"])),
                   "spine_pitch": float(np.median(pitch(sp)))}
            rows.append(row)
        print(g, len(rows), flush=True)
    OUTF.write_text(json.dumps(rows, indent=1))
    summary(rows)


def summary(rows):
    def stat(sel, title):
        if not sel:
            return
        print(f"\n{title}  (n={len(sel)} clips, {len({r['gloss'] for r in sel})} glosses)")
        for k in ("y", "z", "x", "elbow", "shoulder_y"):
            l = np.array([r["lex"][k] for r in sel])
            c = np.array([r["cur"][k] for r in sel])
            f = np.array([r["fold"][k] for r in sel])
            m = np.array([r["cam"][k] for r in sel])
            print(f"  {k:10} lex {np.median(l):+.3f} | cur {np.median(c):+.3f} (diff {np.median(c - l):+.3f}) | "
                  f"fold {np.median(f):+.3f} (diff {np.median(f - l):+.3f}) | cam {np.median(m):+.3f} "
                  f"(diff {np.median(m - l):+.3f})")
    stat(rows, "ALL")
    print()
    for k in ("spine_pitch", "pelvis_cam_pitch", "spine3_cam_pitch"):
        v = np.array([r[k] for r in rows])
        print(f"{k:17} median {np.median(v):+.1f} deg, IQR {np.percentile(v, 25):+.1f}..{np.percentile(v, 75):+.1f}")
    for lc in sorted({r["loc"] for r in rows}):
        stat([r for r in rows if r["loc"] == lc], f"location {lc or '?'}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        summary(json.loads(OUTF.read_text()))
    else:
        main()
