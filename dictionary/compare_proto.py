"""Prototype ASL dictionary: WLASL video -> NLF SMPL-X signs vs the 3D-LEX conversions (ver_04_finetune).

Metric = ver_04_finetune's word-level identification metric, unchanged (scripts/word_metric.py +
word_metric_variants.py): SOKE joints -> body 12 joints pelvis-relative + both hands wrist-relative, rest phases
trimmed (wrist height, frac 0.35), 64 frames, length-normalised DTW, handedness-invariant (min over the mirrored
query). Its calibration on real recordings (SignAvatars-WLASL -> 3D-LEX, 735-way) is the reference ceiling.
Run on the CPU (the GPU belongs to the MaDiS session):
  1. WLASL-NLF clip -> 735 3D-LEX signs: rank of the same gloss (glosses animated in 3D-LEX only);
  2. closed set: WLASL-NLF clip -> the 3D-LEX signs of the prototype glosses only;
  3. WLASL internal consistency: leave-one-out nearest WLASL-NLF clip (other clips of all 23 glosses):
     same gloss? (two other signers of the same sign vs 66 clips of other signs).

    CUDA_VISIBLE_DEVICES= ../ver-01/.venv-signspark/bin/python dictionary/compare_proto.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
sys.path.insert(0, str(ROOT / "ver_04_finetune" / "scripts"))
import word_metric as wm  # noqa: E402
import word_metric_variants as wv  # noqa: E402

W = REPO / "data" / "wlasl" / "streams"
wm._J = wm.sm.SoKeJoints(flat_hand_mean=True, device="cpu")


def dist_matrix(P, G):
    """bt_common.dist_matrix on the CPU: (P, 64, K, 3) x (G, ...) -> length-normalised DTW / 128."""
    Gt = torch.tensor(np.stack(G))
    D = np.zeros((len(P), len(G)))
    for i, p in enumerate(P):
        C = (torch.tensor(p)[None, :, None] - Gt[:, None]).norm(dim=-1).mean(-1)
        D[i] = (wm.bc.dtw_batch(C) / 128).numpy()
    return D


def feats(J):
    return wv.feats(wv.trim(J, 0.35), "full").astype(np.float32)


def ranks(D, own_col):
    return np.array([(D[i] < D[i, c]).sum() + 1 for i, c in enumerate(own_col)])


def summary(r, n):
    return {"n": int(len(r)), "top1": float(np.mean(r == 1)), "top5": float(np.mean(r <= 5)),
            "median_rank": float(np.median(r)), "chance_top1": 1 / n, "chance_median_rank": (n + 1) / 2}


def main():
    lex = sorted(p.name for p in wm.LEX.iterdir() if (p / "meta.json").exists())
    LJ = [wm.stream_joints(*wm.load_streams(wm.LEX / g)) for g in lex]
    LF = [feats(J) for J in LJ]
    clips = sorted(p for p in W.glob("*/*") if (p / "body_rot6d.npy").exists())
    gl = [p.parent.name.replace("_", " ") for p in clips]
    QJ = [wm.stream_joints(*wm.load_streams(p)) for p in clips]
    Q, Qm = [feats(J) for J in QJ], [feats(wm.mirror_joints(J)) for J in QJ]
    D = np.minimum(dist_matrix(Q, LF), dist_matrix(Qm, LF))
    res = {"clips": len(clips), "glosses": sorted(set(gl)), "lex_bank": len(lex)}

    inlex = [k for k, g in enumerate(gl) if g in lex]
    r1 = ranks(D[inlex], [lex.index(gl[k]) for k in inlex])
    res["wlasl_nlf_to_3dlex_735"] = summary(r1, len(lex))
    proto = sorted({gl[k] for k in inlex})
    cols = [lex.index(g) for g in proto]
    r2 = ranks(D[inlex][:, cols], [proto.index(gl[k]) for k in inlex])
    res["wlasl_nlf_to_3dlex_closed"] = summary(r2, len(proto))

    DQ = np.minimum(dist_matrix(Q, Q), dist_matrix(Qm, Q))
    np.fill_diagonal(DQ, np.inf)
    nn = DQ.argmin(1)
    res["wlasl_nlf_loo_nearest_same_gloss"] = float(np.mean([gl[nn[k]] == gl[k] for k in range(len(clips))]))
    res["wlasl_nlf_loo_chance"] = float(np.mean([(sum(g == x for x in gl) - 1) / (len(gl) - 1) for g in gl]))
    res["per_clip"] = {f"{gl[k]}/{clips[k].name}": {"rank_3dlex_735": int(r1[inlex.index(k)]) if k in inlex else None,
                                                   "nearest_wlasl": f"{gl[nn[k]]}/{clips[nn[k]].name}"}
                       for k in range(len(clips))}
    cal = json.loads((ROOT / "ver_04_finetune" / "outputs" / "word_metric_variants.json").read_text())
    res["reference_signavatars_wlasl_to_3dlex"] = cal.get("trim0.35_full", cal)
    out = REPO / "outputs" / "dictionary" / "compare_proto.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_clip"}, indent=1))


if __name__ == "__main__":
    main()
