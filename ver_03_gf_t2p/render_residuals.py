"""Step-3 side-by-side: REAL | baseline (CFG 2.5) | output residual l=2 | embedding residual l=2 | chunks3, candidate 0.

    .venv-signspark/bin/python render_residuals.py
"""
import numpy as np

import render_bestofn as rb
from bt_bestofn import NC, sentences
import bt_common as bc

RD = bc.ROOT / "renders" / "residuals"
sub, test, _, _ = sentences()
Z = np.load(rb.OUTD / "cands.npz", allow_pickle=True)
V = {n: np.load(bc.ROOT / "outputs" / "residuals" / f"{n}.npy", allow_pickle=True)
     for n in ("mean_text", "emb_res_l2", "chunks3")}
from residuals import NV, ortho_rows  # noqa: E402
for i in (len(test) - 1, 3):
    base = Z["feats"][i * NC].astype(np.float32)
    m = V["mean_text"][i * NV].astype(np.float32)
    panels = [rb.panel(bc.gt_feats(test[i]), "REAL recording (LMDB)"),
              rb.panel(base, "baseline CFG 2.5, seed j=0"),
              rb.panel(ortho_rows(m + 2.0 * (base - m)), "output residual l=2 (mean-text sign + 2 x text residual)"),
              rb.panel(V["emb_res_l2"][i * NV].astype(np.float32), "embedding residual l=2"),
              rb.panel(V["chunks3"][i * NV].astype(np.float32), "per-chunk text (3 chunks)")]
    rb.render(RD / test[i]["id"], test[i]["text"], panels, 20, False)
