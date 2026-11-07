"""Body contact for the dictionary signs: torso anchoring (every clip) + ASL-LEX contact snap (coded contact signs).

1. Torso anchor. The spine fold (aslc_build.fold_spine) is right for hands in signing space but wrong for hands on
   the torso: NLF's hand touches NLF's bent chest, and after folding the arm swings while the rendered chest stays
   upright, so the touch point slides down (PLEASE's palm went from 8 to 23 cm below the neck joint, i.e. from
   ASL-LEX's TorsoTop to the stomach). In the unfolded stream the hand sits on the torso exactly as filmed (both
   renderings share the same upright torso). So per arm and frame, the closer the unfolded hand is to the torso
   surface, the more the folded hand is pulled back to the unfolded hand's position and orientation:
   weight 1 within ANCHOR_NEAR of the torso, 0 beyond ANCHOR_FAR, Gaussian-smoothed over time.
2. Contact snap. Signs ASL-LEX 2.0 codes as contact on the body or head (Contact.2.0 = 1, MajorLocation Body/Head)
   still hover (PLEASE +3 cm, 3D-LEX mocap signs +4-10 cm). The dominant hand's closest approach to the coded region
   only (Head -> head/neck surface, Body -> torso surface; the nearest point anywhere put GOOD's hand on the chest
   instead of the chin) is moved onto the surface along its normal over the contact window (frames within 2 cm of
   the closest approach, Gaussian fade sigma 3 frames); skipped if the hand is never within 20 cm.
Both use two-link IK on the stream skeleton (SignSparK streams have identity pelvis/spine), elbow kept in its
swivel plane; anchoring also blends the wrist's global orientation, the snap keeps it (hand moves rigidly).

    .venv-sltk/bin/python dictionary/contact_snap.py [--src final] [GLOSS ...]     (snap in place; meta records it)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aslc_build as ab  # noqa: E402
import contact_eval as ce  # noqa: E402
from compare_proto import wm  # noqa: E402

DICT = ab.OUT
MAX_GAP = 0.20         # head-contact signs hover 11-16 cm at depth scale 0.9 (dict_v3 clips)
ANCHOR_NEAR, ANCHOR_FAR = 0.04, 0.10
TORSOV = np.flatnonzero(np.isin(ce.PART, [0, 3, 6, 9]))                      # pelvis + spine1-3 skin
HEADV = np.flatnonzero(np.isin(ce.PART, [12, 15, 22, 23, 24]))               # neck, head, jaw, eyes
HANDS = {"rh": np.flatnonzero(np.isin(ce.PART, [21, *range(40, 55)])),
         "lh": np.flatnonzero(np.isin(ce.PART, [20, *range(25, 40)]))}
ARM = {"rh": (14, 17, 19, 21), "lh": (13, 16, 18, 20)}                      # collar, shoulder, elbow, wrist


def ik(body, J, t, side, Wt, Gw_new=None):
    """In place on body (T, 10, 3, 3) = local rotations of joints 12-21: put the wrist at Wt (two-link IK, elbow in
    its swivel plane); the wrist's global orientation becomes Gw_new (default: unchanged)."""
    jc, js, je, jw = ARM[side]
    S, E, W = J[t, js], J[t, je], J[t, jw]
    a, b = np.linalg.norm(E - S), np.linalg.norm(W - E)
    d = np.clip(np.linalg.norm(Wt - S), abs(a - b) + 1e-3, 0.995 * (a + b))
    u = (Wt - S) / np.linalg.norm(Wt - S)
    Wt = S + d * u
    x = (a * a - b * b + d * d) / (2 * d)
    r = (E - S) - ((E - S) @ u) * u
    if np.linalg.norm(r) < 1e-6:
        return
    Et = S + x * u + np.sqrt(max(a * a - x * x, 0)) * r / np.linalg.norm(r)
    Gp = body[t, jc - 12]                                                    # spine3 is identity in the stream
    Gs = Gp @ body[t, js - 12]
    Ge = Gs @ body[t, je - 12]
    Gw = Ge @ body[t, jw - 12] if Gw_new is None else Gw_new
    R1 = ab._align(E - S, Et - S)
    Gs2 = R1 @ Gs
    Ge2 = ab._align(R1 @ (W - E), Wt - Et) @ R1 @ Ge
    body[t, js - 12], body[t, je - 12], body[t, jw - 12] = Gp.T @ Gs2, Gs2.T @ Ge2, Ge2.T @ Gw


def global_wrist(body, side):
    jc, js, je, jw = ARM[side]
    return body[:, jc - 12] @ body[:, js - 12] @ body[:, je - 12] @ body[:, jw - 12]


def closest(V, hand, region):
    """Per frame: unsigned closest distance hand -> region vertices."""
    return np.array([float(torch.cdist(torch.tensor(V[t, hand]), torch.tensor(V[t, region])).min())
                     for t in range(len(V))])


def anchor(unfolded, folded):
    """Streams (body60, rh90, lh90) before / after the spine fold -> folded streams with torso contacts restored."""
    Vu, Ju = ce.mesh(*unfolded)
    _, Jf = ce.mesh(*folded)
    T = len(Vu)
    bu, bf = (ab.from6(x[0].reshape(T, -1)) for x in (unfolded, folded))
    out = bf.copy()
    info = {}
    for side in ARM:
        g = closest(Vu, HANDS[side], TORSOV)
        w = gaussian_filter1d(np.clip((ANCHOR_FAR - g) / (ANCHOR_FAR - ANCHOR_NEAR), 0, 1), 2.0, mode="nearest").clip(0, 1)
        Gu, Gf = global_wrist(bu, side), global_wrist(bf, side)
        jw = ARM[side][3]
        for t in np.flatnonzero(w > 1e-3):
            Wt = (1 - w[t]) * Jf[t, jw] + w[t] * Ju[t, jw]
            G = Slerp([0, 1], Rotation.from_matrix(np.stack([Gf[t], Gu[t]])))(w[t]).as_matrix()
            ik(out, Jf, t, side, Wt, G)
        info[side] = round(float(w.mean()), 3)
    return (ab.r6(out), folded[1], folded[2]), info


def gaps(streams, region):
    """Per frame: signed gap (m) of the right hand to the region surface and the outward normal there."""
    V, J = ce.mesh(*streams)
    N = ce.normals(V)
    g, n = np.zeros(len(V)), np.zeros((len(V), 3))
    for t in range(len(V)):
        d, i = torch.cdist(torch.tensor(V[t, HANDS["rh"]]), torch.tensor(V[t, region])).min(1)
        k = int(torch.argmin(d))
        v = region[int(i[k])]
        s = float((V[t, HANDS["rh"][k]] - V[t, v]) @ N[t, v])
        g[t], n[t] = (np.sign(s) if abs(s) > 1e-9 else 1.0) * float(d[k]), N[t, v]
    return g, n, J


def snap(streams, major):
    region = HEADV if major == "Head" else TORSOV
    g, n, J = gaps(streams, region)
    t0 = int(np.argmin(g))
    if abs(g[t0]) > MAX_GAP:                  # far inside = a bad mesh pose (one GOOD clip read -0.56 m), not contact
        return None, {"gap_before": float(g[t0]), "skipped": "hand never near the coded region"}
    win = g <= g[t0] + 0.02
    w = gaussian_filter1d(win.astype(float), 3.0, mode="nearest")
    w = np.maximum(w / max(w[win].min(), 1e-6), win).clip(0, 1)
    shift = -g[t0] * n[t0]
    body = ab.from6(streams[0].reshape(len(g), -1))
    for t in np.flatnonzero(w > 1e-3):
        ik(body, J, t, "rh", J[t, 21] + w[t] * shift)
    out = (ab.r6(body), streams[1], streams[2])
    return out, {"gap_before": float(g[t0]), "gap_after": float(gaps(out, region)[0].min()),
                 "frames_moved": int((w > 0.05).sum()), "shift_m": float(np.linalg.norm(shift))}


def lex_codes():
    """ASL Citizen gloss -> ASL-LEX row (via the CSV's ASL-LEX Code)."""
    lex = {r["Code"]: r for r in csv.DictReader(open(ab.REPO / "data" / "asl_lex" / "signdata.csv", encoding="latin-1"))}
    out = {}
    for s in ("train", "val", "test"):
        for r in csv.DictReader(open(ab.AC / "ASL_Citizen" / "splits" / f"{s}.csv")):
            if r["ASL-LEX Code"] in lex:
                out[r["Gloss"]] = lex[r["ASL-LEX Code"]]
    return out


def is_contact(r):
    return r is not None and r["Contact.2.0"] == "1" and r["MajorLocation.2.0"] in ("Body", "Head")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glosses", nargs="*")
    ap.add_argument("--src", default="final")
    a = ap.parse_args()
    codes = lex_codes()
    dirs = [DICT / a.src / g for g in a.glosses] if a.glosses else sorted(p.parent for p in (DICT / a.src).glob("*/meta.json"))
    for d in dirs:
        meta = json.loads((d / "meta.json").read_text())
        r = codes.get(d.name)
        if "contact_snap" in meta or not is_contact(r):
            continue
        out, info = snap(wm.load_streams(d), r["MajorLocation.2.0"])
        info.update(major=r["MajorLocation.2.0"], minor=r["MinorLocation.2.0"])
        meta["contact_snap"] = info
        if out is not None:
            np.save(d / "body_rot6d.npy", out[0])
        (d / "meta.json").write_text(json.dumps(meta))
        print(f"{d.name:13} {r['MajorLocation.2.0']:4}/{r['MinorLocation.2.0']:10} {info}", flush=True)


if __name__ == "__main__":
    main()
