"""Open-set A/B of whole clip pipelines on the same ASL Citizen clips, against all 735 3D-LEX signs.

  A "fold"  : arm-depth k 1.0 + spine fold                      (dictionary as of the first fold build)
  B "v3"    : arm-depth k 0.9 + spine fold + head level + ASL-LEX contact snap
Scored with ver_04's word metric as is ("full", 12 body points incl. 5 face points) and without face points
("noface": neck, shoulders, elbows, wrists), since 3D-LEX's head is pitched differently. Glosses: all matched
3D-LEX glosses, evenly sampled. Plus elbow and contact gap per clip.

    .venv-sltk/bin/python dictionary/pipeline_eval.py [--max 200] [--per 2]
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
import contact_snap as cs  # noqa: E402
import height_eval as he  # noqa: E402
import height_probe as hp  # noqa: E402
from compare_proto import wm  # noqa: E402

OUTF = ab.REPO / "outputs" / "dictionary" / "pipeline_eval.json"
CONF = {"fold": dict(DEPTH_SCALE=1.0, HEAD_LEVEL=False, CONTACT_SNAP=False),
        "v3": dict(DEPTH_SCALE=0.9, HEAD_LEVEL=True, CONTACT_SNAP=True)}
FULL, NOFACE = list(wm.sm.BODY_IDX), [12, 16, 17, 18, 19, 20, 21]


def bank(idx):
    wm.sm.BODY_IDX[:] = idx
    gl = sorted(p.name for p in wm.LEX.iterdir() if (p / "meta.json").exists())
    return gl, torch.tensor(np.stack([wm.feats_pair(*wm.load_streams(wm.LEX / g))[0] for g in gl]), device=he.DEV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=200)
    ap.add_argument("--per", type=int, default=2)
    a = ap.parse_args()
    banks = {"full": bank(FULL), "noface": bank(NOFACE)}
    gl3 = banks["full"][0]
    lex3 = {hp.norm(g): g for g in gl3}
    codes = cs.lex_codes()
    clips = {}
    for s in ("train", "val", "test"):
        for r in csv.DictReader(open(ab.AC / "ASL_Citizen" / "splits" / f"{s}.csv")):
            stem = Path(r["Video file"]).stem
            if hp.norm(r["Gloss"]) in lex3 and (ab.FEATS / "nlf" / f"{stem}.h5").exists() and \
                    (ab.FEATS / "wilor" / f"{stem}.h5").exists():
                clips.setdefault(r["Gloss"], []).append(stem)
    pick = sorted(clips)[:: max(1, len(clips) // a.max)][: a.max]
    print(f"{len(clips)} matched glosses, evaluating {len(pick)} x {a.per}", flush=True)
    fast = torch.load(ab.FEATS / "fast.pt", weights_only=False)
    st = np.load(ab.REPO / "outputs" / "static_How2Sign.npz")
    ab.SPINE_FOLD, ab.TORSO_ANCHOR = True, False
    rows = []
    for n, g in enumerate(pick):
        own = gl3.index(lex3[hp.norm(g)])
        for stem in clips[g][: a.per]:
            sg = fast.get(stem)
            row = {"gloss": g, "stem": stem, "contact": cs.is_contact(codes.get(g))}
            for name, conf in CONF.items():
                for k, v in conf.items():
                    setattr(ab, k, v)
                s, meta = ab.load_clip(stem, sg.numpy() if sg is not None else None, st["rh"], st["lh_rframe"])
                if s is None:
                    break
                s = ab.apply_contact(s, meta, g)
                streams = (ab.r6(s["body"][:, 11:21]), ab.r6(s["rh"]), ab.r6(s["lh_model"]))
                m = {"elbow": hp.measure(*streams)["elbow"]}
                for bk, (_, Gt) in banks.items():
                    wm.sm.BODY_IDX[:] = FULL if bk == "full" else NOFACE
                    f, fm = wm.feats_pair(*streams)
                    D = np.minimum(he.dists(f, Gt), he.dists(fm, Gt))
                    m[bk] = int((D < D[own]).sum() + 1)
                row[name] = m
            else:
                rows.append(row)
        if n % 20 == 0:
            print(n, g, len(rows), flush=True)
    OUTF.write_text(json.dumps(rows, indent=1))
    for bk in ("full", "noface"):
        for name in CONF:
            r = np.array([x[name][bk] for x in rows])
            print(f"{bk:6} {name:5} n={len(r)} top1 {np.mean(r == 1):.3f} top5 {np.mean(r <= 5):.3f} "
                  f"top20 {np.mean(r <= 20):.3f} median rank {np.median(r):.0f}")
        d = np.array([x["v3"][bk] - x["fold"][bk] for x in rows])
        print(f"{bk:6} v3 better on {np.mean(d < 0):.2f} of clips, worse on {np.mean(d > 0):.2f}")
    print(f"elbow median fold {np.median([x['fold']['elbow'] for x in rows]):.1f}  v3 {np.median([x['v3']['elbow'] for x in rows]):.1f}")


if __name__ == "__main__":
    main()
