"""Evaluate outputs/vq_roundtrip.npz (made by bench_b_madis.py): how much a MaDiS VQ round trip changes 3D-LEX signs
vs How2Sign references, in joints (SOKE joints, metres), and whether a round-tripped 3D-LEX sign is still
identified as itself among the 735 originals (word_metric, trim 0.35, full features).
    ../ver-01/.venv-signspark/bin/python scripts/vq_eval.py"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ver_03_madis"))
import word_metric as wm  # noqa: E402
import word_metric_variants as wv  # noqa: E402
from convert_madis import convert  # noqa: E402

FN = sys.argv[1] if len(sys.argv) > 1 else "vq_roundtrip.npz"
z = np.load(wm.HERE / "outputs" / FN)
M, S = z["mean"], z["std"]


def J(fn):
    return wm.stream_joints(*convert(fn.astype(np.float64) * S + M))


def err(a, b):
    w = [20, 21]
    pr = lambda J: J["body"][:, wm.sm.BODY_IDX] - J["body"][:, :1]  # noqa: E731
    hand = lambda J: np.concatenate([J["lhand"] - J["lhand"][:, :1], J["rhand"] - J["rhand"][:, :1]], 1)  # noqa: E731
    return {"wrist_mm": float(np.linalg.norm(a["body"][:, w] - b["body"][:, w], axis=-1).mean() * 1000),
            "body12_mm": float(np.linalg.norm(pr(a) - pr(b), axis=-1).mean() * 1000),
            "hand_mm": float(np.linalg.norm(hand(a) - hand(b), axis=-1).mean() * 1000)}


res = {}
for src in [s for s in ("lex", "h2s") if any(k.startswith(s + "_") for k in z.files)]:
    names = sorted({k[len(src) + 1:-4] for k in z.files if k.startswith(src + "_")})
    pairs = [(J(z[f"{src}_{n}_ref"]), J(z[f"{src}_{n}_dec"])) for n in names]
    e = [err(a, b) for a, b in pairs]
    res[src] = {k: float(np.mean([x[k] for x in e])) for k in e[0]} | {"n": len(e)}
    if src == "lex":
        F = np.stack([wv.feats(wv.trim(a, 0.35), "full") for a, _ in pairs]).astype(np.float32)
        q = [(wv.feats(wv.trim(b, 0.35), "full"), wv.feats(wv.trim(wm.mirror_joints(b), 0.35), "full")) for _, b in pairs]
        _, r = wm.rank_queries(q, names, names, F)
        res["lex_roundtrip_identity_retrieval"] = r
        worst = sorted(zip(names, e), key=lambda t: -t[1]["hand_mm"])[:5]
        res["lex_worst_hand"] = {n: x for n, x in worst}
(wm.HERE / "outputs" / FN.replace("vq_roundtrip", "vq_eval").replace(".npz", ".json")).write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
