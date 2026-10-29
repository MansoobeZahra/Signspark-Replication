"""Why do re-encoded How2Sign references not reproduce the released tokens? Per part: agreement, and feature error
of (decode released tokens) vs (decode our tokens) against the reference, on 50 test clips.
    ../ver_03_madis/.venv-madis/bin/python scripts/vq_debug.py"""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_b_madis as bb  # noqa: E402

model, _ = bb.gm.load(bb.DEV)
items = bb.h2s_items(model)[:50]
agree, e_rel, e_our, e_rel_shift = [], [], [], []
with torch.no_grad():
    for it in items:
        x = torch.tensor(it["ref"], device=bb.DEV)[None]
        parts = bb.split(x)
        vs = (model.vae, model.hand_vae, model.rhand_vae)
        rel = torch.tensor(it["tok"], device=bb.DEV)
        a, er, eo = [], [], []
        for k, (v, p) in enumerate(zip(vs, parts)):
            ours = v.encode(p)[0][0]
            a.append(float((ours == rel[:, k]).float().mean()))
            er.append(float((v.decode(rel[:, k]) - p).abs().mean()))
            eo.append(float((v.decode(ours) - p).abs().mean()))
        agree.append(a); e_rel.append(er); e_our.append(eo)
print("token agreement body/lh/rh", np.mean(agree, 0).round(3))
print("feature L1 (normalised units) decode(RELEASED) vs ref", np.mean(e_rel, 0).round(3))
print("feature L1 decode(OURS) vs ref                ", np.mean(e_our, 0).round(3))
print("codebook buffer vs vae codebook equal:",
      [bool(torch.equal(getattr(model.lm.language_model, f"{n}_codebook").float().cpu(), v.quantizer.codebook.float().cpu()))
       if hasattr(model.lm.language_model, f"{n}_codebook") and hasattr(v.quantizer, "codebook") else "n/a"
       for n, v in (("body", model.vae), ("lhand", model.hand_vae), ("rhand", model.rhand_vae))])
