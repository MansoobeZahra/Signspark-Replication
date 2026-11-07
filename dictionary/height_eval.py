"""Which torso/height correction makes ASL Citizen signs closest to the real ones? Open-set retrieval against 3D-LEX.

Every ASL Citizen clip whose gloss is also in 3D-LEX is built once (aslc_build.load_clip) and rendered as several
variants of the same stream; each variant is ranked against all 735 3D-LEX signs with ver_04's word metric
(DTW over pelvis-relative body + wrist-relative hands, trim 0.35, handedness-invariant). Higher top-k = closer to
the mocap signs. Reference: real SignAvatars-WLASL recordings reach ~0.55 top-5 on the same bank (PAPER_AUDIT §6).
Variants:
  noarm : arm-depth fix off (the old dictionary)
  cur   : arm-depth fix on (current default)
  fold  : + spine1-3 rotation (relative to the pelvis) folded into neck and collars
  cam   : + only the forward pitch of spine3 relative to the camera vertical folded in (constant per clip)
Output: outputs/dictionary/height_eval.json

    .venv-sltk/bin/python dictionary/height_eval.py [--per 2] [--max 400]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aslc_build as ab  # noqa: E402
import height_probe as hp  # noqa: E402
from compare_proto import wm  # noqa: E402

OUTF = ab.REPO / "outputs" / "dictionary" / "height_eval.json"


DEV = "cuda" if torch.cuda.is_available() else "cpu"


def dists(q, Gt):
    """Length-normalised DTW of one query (64, K, 3) to the bank (G, 64, K, 3) on DEV (= compare_proto.dist_matrix)."""
    C = (torch.tensor(q, device=DEV)[None, :, None] - Gt[:, None]).norm(dim=-1).mean(-1)
    return (wm.bc.dtw_batch(C) / 128).cpu().numpy()


def variants(stem, seg, st):
    out = {}
    ab.SPINE_FOLD = False                       # variants below fold by hand (fold is the build default since)
    ab.ARM_DEPTH = False
    s, _ = ab.load_clip(stem, seg, st["rh"], st["lh_rframe"])
    if s is None:
        return None
    out["noarm"] = s["body"]
    ab.ARM_DEPTH = True
    s, _ = ab.load_clip(stem, seg, st["rh"], st["lh_rframe"])
    b = s["body"]
    out["cur"], out["fold"] = b, hp.fold(b)
    out["cam"] = hp.fold(b, hp.rx(hp.cam_pitch(stem)[1]))
    return {k: (ab.r6(v[:, 11:21]), ab.r6(s["rh"]), ab.r6(s["lh_model"])) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=2)
    ap.add_argument("--max", type=int, default=400)
    a = ap.parse_args()
    gl, F = wm.lex_bank()
    lex = {hp.norm(g): g for g in gl}
    import csv
    clips = {}
    for s in ("train", "val", "test"):
        for r in csv.DictReader(open(ab.AC / "ASL_Citizen" / "splits" / f"{s}.csv")):
            stem = Path(r["Video file"]).stem
            if hp.norm(r["Gloss"]) in lex and (ab.FEATS / "nlf" / f"{stem}.h5").exists() and \
                    (ab.FEATS / "wilor" / f"{stem}.h5").exists():
                clips.setdefault(r["Gloss"], []).append(stem)
    glosses = sorted(clips)[:: max(1, len(clips) // a.max)][: a.max]
    print(f"{len(clips)} matched glosses, evaluating {len(glosses)} x {a.per} clips vs {len(gl)} 3D-LEX signs", flush=True)
    fast = torch.load(ab.FEATS / "fast.pt", weights_only=False)
    st = np.load(ab.REPO / "outputs" / "static_How2Sign.npz")
    Gt = torch.tensor(F, device=DEV)
    ranks, rows = {}, []
    for i, g in enumerate(glosses):
        own = gl.index(lex[hp.norm(g)])
        for stem in clips[g][: a.per]:
            sg = fast.get(stem)
            try:
                V = variants(stem, sg.numpy() if sg is not None else None, st)
            except Exception as e:
                print(g, stem, type(e).__name__, e, flush=True)
                continue
            if V is None:
                continue
            row = {"gloss": g, "stem": stem}
            for k, streams in V.items():
                f, fm = wm.feats_pair(*streams)
                D = np.minimum(dists(f, Gt), dists(fm, Gt))
                row[k] = int((D < D[own]).sum() + 1)
                ranks.setdefault(k, []).append(row[k])
            rows.append(row)
        if i % 20 == 0:
            print(i, g, {k: f"top5 {np.mean(np.array(v) <= 5):.3f}" for k, v in ranks.items()}, flush=True)
    res = {k: {"n": len(v), "top1": float(np.mean(np.array(v) == 1)), "top5": float(np.mean(np.array(v) <= 5)),
               "top20": float(np.mean(np.array(v) <= 20)), "median_rank": float(np.median(v))}
           for k, v in ranks.items()}
    OUTF.write_text(json.dumps({"summary": res, "rows": rows}, indent=1))
    for k, v in res.items():
        print(f"{k:6} n={v['n']}  top1 {v['top1']:.3f}  top5 {v['top5']:.3f}  top20 {v['top20']:.3f}  "
              f"median rank {v['median_rank']:.0f}")


if __name__ == "__main__":
    main()
