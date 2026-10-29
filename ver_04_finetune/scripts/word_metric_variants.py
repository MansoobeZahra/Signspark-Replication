"""Calibration of word-level metric variants on REAL recordings (WLASL clip -> 3D-LEX sign of the same gloss,
735-way). Variants: trimming the rest phases (wrists below a height threshold), features (body+hands as in the
sentence retrieval / hands only / wrist trajectories only). Writes outputs/word_metric_variants.json."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import word_metric as wm  # noqa: E402

WR = (20, 21)                      # SMPL-X left/right wrist in J['body']


def trim(J, frac):
    if frac is None:
        return J
    y = J["body"][:, WR, 1] - J["body"][:, :1, 1]          # wrist height relative to pelvis
    h = y.max(1)
    thr = h.min() + frac * (h.max() - h.min())
    idx = np.where(h >= thr)[0]
    a, b = idx[0], idx[-1] + 1
    if b - a < 4:
        return J
    return {k: v[a:b] for k, v in J.items()}


def feats(J, kind):
    import bt_common as bc
    if kind == "full":
        return bc.retrieval_feats(J)
    b = J["body"][:, wm.sm.BODY_IDX] - J["body"][:, :1]
    if kind == "hands":
        x = np.concatenate([J["lhand"] - J["lhand"][:, :1], J["rhand"] - J["rhand"][:, :1]], 1)
    elif kind == "wrists":
        x = b[:, [5, 6]]                                     # BODY_IDX[5:7] = wrists 20, 21
    elif kind == "full_w3":                                  # wrists weighted x3 (location/movement emphasis)
        x = np.concatenate([b, np.repeat(b[:, [5, 6]], 20, 1), J["lhand"] - J["lhand"][:, :1], J["rhand"] - J["rhand"][:, :1]], 1)
    t = np.linspace(0, len(x) - 1, 64)
    return np.stack([np.stack([np.interp(t, np.arange(len(x)), x[:, k, c]) for c in range(3)], -1) for k in range(x.shape[1])], 1)


def main():
    lex = sorted(p.name for p in wm.LEX.iterdir() if (p / "meta.json").exists())
    ov = json.loads((wm.HERE / "outputs/vocab_overlap.json").read_text())["lex_anim_and_wlasl"]
    g2i = json.loads((wm.ROOT / "ver_03_sr_t2p/data/meta/gloss_to_ids.json").read_text())
    conv = wm.ROOT / "ver_03_sr_t2p/data/converted"
    LJ = [wm.stream_joints(*wm.load_streams(wm.LEX / g)) for g in lex]
    Q, own = [], []
    for g in ov:
        for i in g2i[g]:
            if (conv / i).exists():
                Q.append(wm.stream_joints(*wm.load_streams(conv / i))); own.append(g)
    res = {}
    for fr in (None, 0.2, 0.35):
        for kind in ("full", "hands", "wrists"):
            F = [feats(trim(J, fr), kind).astype(np.float32) for J in LJ]
            q = [(feats(trim(J, fr), kind), feats(trim(wm.mirror_joints(J), fr), kind)) for J in Q]
            D, r = wm.rank_queries(q, own, lex, np.stack(F))
            ranks = np.array([(D[k] < D[k, lex.index(o)]).sum() + 1 for k, o in enumerate(own)])
            best = [ranks[[k for k, o in enumerate(own) if o == g]].min() for g in ov]
            key = f"trim{fr}_{kind}"
            res[key] = {"clip_top1": r["top1"], "clip_top5": r["top5"], "clip_median_rank": r["median_rank"],
                        "gloss_bestclip_top1": float(np.mean(np.array(best) == 1)), "gloss_bestclip_top5": float(np.mean(np.array(best) <= 5))}
            print(key, {k: round(v, 3) for k, v in res[key].items()}, flush=True)
    (wm.HERE / "outputs/word_metric_variants.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
