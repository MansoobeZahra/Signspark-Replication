"""Calibrate the arm-depth scale and the spine-fold fraction on ASL-LEX contact signs (linguistic ground truth,
independent of 3D-LEX's signer and of its SMPL-X conversion).

ASL-LEX 2.0 codes, per sign, whether the dominant hand touches the body (Contact.2.0) and where
(MajorLocation Body/Head, MinorLocation TorsoTop / Clavicle / Chin / Forehead ...). For every contact sign on the
body or head the avatar's dominant hand must actually reach that surface:
  * gap (m): closest approach over the sign of the right-hand surface vertices to the torso/neck/head surface,
    signed by the surface normal (> 0 floats in front, < 0 inside the body). Should be ~0. Too-bent elbows put
    the hand too far back (negative); too much forward correction makes it float (positive).
  * contact height (m) of the touched body point below the neck joint (Body signs), to compare with ASL-LEX's
    minor location (rest SMPL-X: neck joint 0, spine3 -0.16, spine2 -0.22, spine1 -0.34, pelvis -0.45; the upper
    chest "TorsoTop" spans roughly -0.06..-0.20).
Variants of the same clips: depth scale k (aslc_build.DEPTH_SCALE: x WiLoR hand depth, < 1 moves hands toward
the camera) x fold fraction a (aslc_build.fold_spine alpha). 3D-LEX signs of the same glosses are measured too, as
a mocap reference for the metric itself. Glosses also in 3D-LEX get the open-set rank against its 735 signs.

    .venv-sltk/bin/python dictionary/contact_eval.py [--per 2] [--max 150]
Output: outputs/dictionary/contact_eval.json + printed table.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aslc_build as ab  # noqa: E402
import height_eval as he  # noqa: E402
import height_probe as hp  # noqa: E402
from compare_proto import wm, wv  # noqa: E402

import soke_metric as sm  # noqa: E402  (on sys.path via word_metric)

OUTF = ab.REPO / "outputs" / "dictionary" / "contact_eval.json"
KS = (1.1, 1.0, 0.9, 0.8, 0.7)
ALPHAS = (0.6, 0.8, 1.0)
LAYER = wm._J.layer
PART = LAYER.lbs_weights.detach().cpu().numpy().argmax(1)
BODYV = np.flatnonzero(np.isin(PART, [0, 3, 6, 9, 12, 13, 14, 15, 22, 23, 24]))   # torso, collars, neck, head
HANDV = np.flatnonzero(np.isin(PART, [21, *range(40, 55)]))                        # right hand surface
FACES = LAYER.faces_tensor.cpu().numpy() if hasattr(LAYER, "faces_tensor") else LAYER.faces.astype(np.int64)


def mesh(body60, rh, lh):
    """Posed SMPL-X vertices + joints exactly as word_metric.stream_joints poses them."""
    T = len(body60)
    body = np.zeros((T, 21, 3))
    body[:, 11:21] = ab.Rotation.from_matrix(ab.from6(body60.reshape(T, -1)).reshape(-1, 3, 3)).as_rotvec().reshape(T, 10, 3)
    lha, rha = wm.hand_pose_axis_angle("left", lh), wm.hand_pose_axis_angle("right", rh)
    f = lambda a: torch.as_tensor(np.asarray(a).reshape(T, -1), dtype=torch.float32)  # noqa: E731
    z = torch.zeros(T, 3)
    with torch.no_grad():
        o = LAYER(betas=torch.tensor([sm.SOKE_BETAS]).repeat(T, 1), global_orient=z, body_pose=f(body),
                  left_hand_pose=f(lha), right_hand_pose=f(rha), jaw_pose=z, leye_pose=z, reye_pose=z,
                  expression=torch.zeros(T, 10))
    return o.vertices.numpy(), o.joints.numpy()


def normals(V):
    n = np.zeros_like(V)
    tri = V[:, FACES]
    fn = np.cross(tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0])
    for k in range(3):
        np.add.at(n, (slice(None), FACES[:, k]), fn)
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)


def contact(streams):
    """-> gap (m, signed), contact height below the neck joint (m), contact height relative to the head joint."""
    V, Jt = mesh(*streams)
    h = (Jt[:, wv.WR, 1] - Jt[:, :1, 1]).max(1)                     # = word_metric_variants.trim(J, 0.35)
    idx = np.flatnonzero(h >= h.min() + 0.35 * (h.max() - h.min()))
    if idx[-1] + 1 - idx[0] >= 4:
        V, Jt = V[idx[0]: idx[-1] + 1], Jt[idx[0]: idx[-1] + 1]
    N = normals(V)
    best = (np.inf, None, None)
    for t in range(len(V)):
        B, H = torch.tensor(V[t, BODYV]), torch.tensor(V[t, HANDV])
        d, i = torch.cdist(H, B).min(1)
        k = int(torch.argmin(d))
        v = BODYV[int(i[k])]
        s = float((V[t, HANDV[k]] - V[t, v]) @ N[t, v])
        g = np.sign(s) * float(d[k]) if abs(s) > 1e-9 else float(d[k])
        if g < best[0]:
            best = (g, V[t, v, 1] - Jt[t, 12, 1], V[t, v, 1] - Jt[t, 15, 1])
    el = hp.measure(*streams)["elbow"]
    return {"gap": best[0], "h_neck": float(best[1]), "h_head": float(best[2]), "elbow": el}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=2)
    ap.add_argument("--max", type=int, default=150, help="glosses per major location (Body, Head)")
    a = ap.parse_args()
    lexr = {}
    for r in csv.DictReader(open(ab.REPO / "data" / "asl_lex" / "signdata.csv", encoding="latin-1")):
        lexr[r["Code"]] = r
    gl3, F = wm.lex_bank()
    lex3 = {hp.norm(g): g for g in gl3}
    Gt = torch.tensor(F, device=he.DEV)
    clips = {}
    for s in ("train", "val", "test"):
        for r in csv.DictReader(open(ab.AC / "ASL_Citizen" / "splits" / f"{s}.csv")):
            x = lexr.get(r["ASL-LEX Code"])
            stem = Path(r["Video file"]).stem
            if x is None or x["Contact.2.0"] != "1" or x["MajorLocation.2.0"] not in ("Body", "Head"):
                continue
            if (ab.FEATS / "nlf" / f"{stem}.h5").exists() and (ab.FEATS / "wilor" / f"{stem}.h5").exists():
                clips.setdefault(r["Gloss"], {"stems": [], "major": x["MajorLocation.2.0"],
                                              "minor": x["MinorLocation.2.0"]})["stems"].append(stem)
    pick = []
    for maj in ("Body", "Head"):
        g = sorted(k for k, v in clips.items() if v["major"] == maj)
        pick += g[:: max(1, len(g) // a.max)][: a.max]
    print(f"{len(clips)} contact glosses with features; evaluating {len(pick)}", flush=True)
    fast = torch.load(ab.FEATS / "fast.pt", weights_only=False)
    st = np.load(ab.REPO / "outputs" / "static_How2Sign.npz")
    rows = []
    ab.SPINE_FOLD = False
    for n, g in enumerate(pick):
        e = clips[g]
        lexname = lex3.get(hp.norm(g))
        ref = contact(wm.load_streams(wm.LEX / lexname)) if lexname else None
        for stem in e["stems"][: a.per]:
            sg = fast.get(stem)
            sg = sg.numpy() if sg is not None else None
            row = {"gloss": g, "stem": stem, "major": e["major"], "minor": e["minor"], "lex": ref, "v": {}}
            try:
                for k in KS:
                    ab.ARM_DEPTH, ab.DEPTH_SCALE = True, k
                    s, _ = ab.load_clip(stem, sg, st["rh"], st["lh_rframe"])
                    if s is None:
                        raise ValueError("no clip")
                    for al in ALPHAS:
                        b = ab.fold_spine(s["body"], al)
                        streams = (ab.r6(b[:, 11:21]), ab.r6(s["rh"]), ab.r6(s["lh_model"]))
                        m = contact(streams)
                        if lexname:
                            f, fm = wm.feats_pair(*streams)
                            D = np.minimum(he.dists(f, Gt), he.dists(fm, Gt))
                            m["rank"] = int((D < D[gl3.index(lexname)]).sum() + 1)
                        row["v"][f"k{k}_a{al}"] = m
            except Exception as ex:
                print(g, stem, type(ex).__name__, ex, flush=True)
                continue
            rows.append(row)
        if n % 10 == 0:
            print(n, g, len(rows), flush=True)
            OUTF.write_text(json.dumps(rows))
    OUTF.write_text(json.dumps(rows, indent=1))
    summary(rows)


def summary(rows):
    keys = list(rows[0]["v"])
    print(f"\n{len(rows)} clips. gap = closest approach of the hand to the body surface (m, + = floating, - = inside)")
    print(f"{'variant':12} {'|gap| med':>9} {'gap med':>8} {'inside%':>7} {'>3cm%':>6} {'elbow':>6} "
          f"{'TorsoTop h':>10} {'Body h':>7} {'top5':>5} {'medrank':>7}")
    lexrows = [r for r in rows if r["lex"]]
    L = np.array([r["lex"]["gap"] for r in lexrows])
    lt = [r["lex"]["h_neck"] for r in lexrows if r["minor"] == "TorsoTop"]
    print(f"{'3D-LEX ref':12} {np.median(np.abs(L)):9.3f} {np.median(L):+8.3f} {np.mean(L < -0.01) * 100:7.0f} "
          f"{np.mean(L > 0.03) * 100:6.0f} {np.median([r['lex']['elbow'] for r in lexrows]):6.1f} "
          f"{np.median(lt) if lt else np.nan:+10.3f}")
    for k in keys:
        g = np.array([r["v"][k]["gap"] for r in rows])
        tt = [r["v"][k]["h_neck"] for r in rows if r["minor"] == "TorsoTop"]
        bh = [r["v"][k]["h_neck"] for r in rows if r["major"] == "Body"]
        rk = np.array([r["v"][k]["rank"] for r in rows if "rank" in r["v"][k]])
        print(f"{k:12} {np.median(np.abs(g)):9.3f} {np.median(g):+8.3f} {np.mean(g < -0.01) * 100:7.0f} "
              f"{np.mean(g > 0.03) * 100:6.0f} {np.median([r['v'][k]['elbow'] for r in rows]):6.1f} "
              f"{np.median(tt):+10.3f} {np.median(bh):+7.3f} {np.mean(rk <= 5):5.3f} {np.median(rk):7.0f}")
    print("\nper ASL-LEX minor location (Body), contact height below neck joint, median: lex-ref | " +
          " | ".join(keys))
    for mn in sorted({r["minor"] for r in rows if r["major"] == "Body"}):
        sel = [r for r in rows if r["minor"] == mn]
        lx = [r["lex"]["h_neck"] for r in sel if r["lex"]]
        print(f"  {mn:12} n={len(sel):3} {np.median(lx) if lx else np.nan:+.3f} | " +
              " ".join(f"{np.median([r['v'][k]['h_neck'] for r in sel]):+.3f}" for k in keys))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        summary(json.loads(OUTF.read_text()))
    else:
        main()
