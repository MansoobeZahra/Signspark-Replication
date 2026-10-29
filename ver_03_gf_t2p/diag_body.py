"""Which body joints make text-only SignSparK's body DTW-JPE worse than a motionless pose? (step-1 diagnosis)"""
import numpy as np
import soke_metric as sm
from eval_t2p import load_subset, smplx_in, OUT

clips = load_subset(200)
fk = sm.SoKeJoints()
SUB = {"arms (neck, shoulders, elbows, wrists)": sm.BODY_IDX[:7], "head (ears, eyes, nose)": sm.BODY_IDX[7:],
       "wrists only": [20, 21]}
rows = {}
for name in ("ours", "static_mean"):
    P = np.load(OUT / f"{name}_pred.npz")
    acc = {k: [] for k in SUB}
    pitch = []
    for k, c in enumerate(clips):
        n = int(P["lengths"][k])
        gt = fk(*smplx_in(c["body"], c["rh"], c["lh"], False))
        pr = fk(*smplx_in(P["body"][k, 0, :n], P["hand"][k, 0, :n], P["hand"][k, 1, :n], True))
        for s, idx in SUB.items():
            acc[s].append(sm.dtw_fast(sm.pair_costs(pr["body"], gt["body"], idx, 0)))
        # head direction: nose relative to neck, pred minus GT mean (mm, y = up, z = forward)
        pitch.append((pr["body"][:, 55] - pr["body"][:, 12]).mean(0) - (gt["body"][:, 55] - gt["body"][:, 12]).mean(0))
    rows[name] = {s: np.mean(v) for s, v in acc.items()}
    print(f"{name:12}", "  ".join(f"{s}: {v:.2f}" for s, v in rows[name].items()),
          "| mean nose-neck offset vs GT (mm, x y z):", (np.mean(pitch, 0) * 1000).round(1))
