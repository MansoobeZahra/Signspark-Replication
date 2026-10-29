"""Follow-up to vq_debug.py: baselines (mean pose = zeros in normalised units; the clip's own temporal mean) and
every assignment of released token columns to the three VAEs; also a time-reversed check. 50 test clips.
    ../ver_03_madis/.venv-madis/bin/python scripts/vq_debug2.py"""
import itertools, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_b_madis as bb  # noqa: E402

model, _ = bb.gm.load(bb.DEV)
items = bb.h2s_items(model)[:50]
vs = (model.vae, model.hand_vae, model.rhand_vae)
names = ("body", "lh", "rh")
acc = {}
with torch.no_grad():
    for it in items:
        parts = bb.split(torch.tensor(it["ref"], device=bb.DEV)[None])
        rel = torch.tensor(it["tok"], device=bb.DEV)
        for k, p in enumerate(parts):
            acc.setdefault(("zeros", k), []).append(float(p.abs().mean()))
            acc.setdefault(("clipmean", k), []).append(float((p - p.mean(1, keepdim=True)).abs().mean()))
            for j in range(3):
                vocab = vs[k].quantizer.codebook.shape[0] if hasattr(vs[k].quantizer, "codebook") else 10 ** 9
                if int(rel[:, j].max()) < vocab:
                    acc.setdefault((f"released col{j}", k), []).append(float((vs[k].decode(rel[:, j]) - p).abs().mean()))
                    acc.setdefault((f"released col{j} reversed", k), []).append(float((vs[k].decode(rel[:, j].flip(0)) - p).abs().mean()))
for k in range(3):
    print(names[k], {key: round(float(np.mean(v)), 3) for (key, kk), v in sorted(acc.items()) if kk == k})
print("released max code per column", np.max([np.max(it["tok"], 0) for it in items], 0),
      "codebook sizes", [v.quantizer.codebook.shape[0] for v in vs])
