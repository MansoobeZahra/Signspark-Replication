"""Arm-depth scale for signs made in the air (ASL-LEX MajorLocation Neutral/Hand, no body contact).

contact_eval covers contact signs (k = 1.1 touches, k < 1 lifts the hand off). Here: glosses also in 3D-LEX whose
ASL-LEX coding is Neutral/Hand without contact; per clip, depth scale k (aslc_build.DEPTH_SCALE) with the spine fold
on, measured against the 3D-LEX sign of the same gloss: open-set rank among 735 signs, elbow angle, wrist reach
(distance from the shoulder) and wrist relative to the shoulder.

    .venv-sltk/bin/python dictionary/depth_eval.py [--max 150] [--per 1]
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
from compare_proto import wm  # noqa: E402

KS = (1.2, 1.1, 1.0, 0.9, 0.8)
OUTF = ab.REPO / "outputs" / "dictionary" / "depth_eval.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=150)
    ap.add_argument("--per", type=int, default=1)
    a = ap.parse_args()
    lexr = {r["Code"]: r for r in csv.DictReader(open(ab.REPO / "data" / "asl_lex" / "signdata.csv", encoding="latin-1"))}
    gl3, F = wm.lex_bank()
    lex3 = {hp.norm(g): g for g in gl3}
    Gt = torch.tensor(F, device=he.DEV)
    clips = {}
    for s in ("train", "val", "test"):
        for r in csv.DictReader(open(ab.AC / "ASL_Citizen" / "splits" / f"{s}.csv")):
            x, stem = lexr.get(r["ASL-LEX Code"]), Path(r["Video file"]).stem
            if x is None or x["Contact.2.0"] != "0" or x["MajorLocation.2.0"] not in ("Neutral", "Hand") \
                    or hp.norm(r["Gloss"]) not in lex3:
                continue
            if (ab.FEATS / "nlf" / f"{stem}.h5").exists() and (ab.FEATS / "wilor" / f"{stem}.h5").exists():
                clips.setdefault(r["Gloss"], []).append(stem)
    pick = sorted(clips)[:: max(1, len(clips) // a.max)][: a.max]
    print(f"{len(clips)} non-contact Neutral/Hand glosses in 3D-LEX; evaluating {len(pick)}", flush=True)
    fast = torch.load(ab.FEATS / "fast.pt", weights_only=False)
    st = np.load(ab.REPO / "outputs" / "static_How2Sign.npz")
    ab.SPINE_FOLD, ab.TORSO_ANCHOR = True, False
    rows = []
    for n, g in enumerate(pick):
        name = lex3[hp.norm(g)]
        L = hp.measure(*wm.load_streams(wm.LEX / name))
        for stem in clips[g][: a.per]:
            sg = fast.get(stem)
            row = {"gloss": g, "stem": stem, "lex": L}
            for k in KS:
                ab.DEPTH_SCALE = k
                s, _ = ab.load_clip(stem, sg.numpy() if sg is not None else None, st["rh"], st["lh_rframe"])
                if s is None:
                    break
                streams = (ab.r6(s["body"][:, 11:21]), ab.r6(s["rh"]), ab.r6(s["lh_model"]))
                m = hp.measure(*streams)
                f, fm = wm.feats_pair(*streams)
                D = np.minimum(he.dists(f, Gt), he.dists(fm, Gt))
                m["rank"] = int((D < D[gl3.index(name)]).sum() + 1)
                row[str(k)] = m
            else:
                rows.append(row)
        if n % 20 == 0:
            print(n, g, len(rows), flush=True)
    OUTF.write_text(json.dumps(rows, indent=1))
    reach = lambda m: np.sqrt(m["x"] ** 2 + m["y"] ** 2 + m["z"] ** 2)  # noqa: E731
    print(f"\n{len(rows)} clips  | 3D-LEX: elbow {np.median([r['lex']['elbow'] for r in rows]):.1f}  reach "
          f"{np.median([reach(r['lex']) for r in rows]):.3f}  y {np.median([r['lex']['y'] for r in rows]):+.3f}  "
          f"z {np.median([r['lex']['z'] for r in rows]):+.3f}")
    for k in KS:
        m = [r[str(k)] for r in rows]
        rk = np.array([x["rank"] for x in m])
        print(f"k={k:<4} top1 {np.mean(rk == 1):.3f} top5 {np.mean(rk <= 5):.3f} top20 {np.mean(rk <= 20):.3f} "
              f"med rank {np.median(rk):5.0f} | elbow {np.median([x['elbow'] for x in m]):5.1f} (diff "
              f"{np.median([x['elbow'] - r['lex']['elbow'] for x, r in zip(m, rows)]):+5.1f})  reach "
              f"{np.median([reach(x) for x in m]):.3f}  y {np.median([x['y'] for x in m]):+.3f}  "
              f"z {np.median([x['z'] for x in m]):+.3f}")


if __name__ == "__main__":
    main()
