"""Distance of word-level data from SignSparK's How2Sign fittings, per stream: mean |z| of rot6D features against the
How2Sign test LMDB statistics (500 clips), per-dimension std floored at 0.05 (some rot6D components are nearly
constant in the fittings). Reference: How2Sign itself. Writes outputs/domain_gap.json.
    ../ver-01/.venv-signspark/bin/python scripts/domain_gap.py"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]; R = HERE.parent
sys.path.insert(0, str(R / "ver_03_gf_t2p"))
import eval_t2p as ev  # noqa: E402
c = ev.load_subset(500)
H = {k: np.concatenate([x[k] for x in c]) for k in ("body", "rh", "lh")}
mu = {k: v.mean(0) for k, v in H.items()}
sd = {k: np.maximum(v.std(0), 0.05) for k, v in H.items()}
F = {"body": "body_rot6d", "rh": "hand_rot6d", "lh": "hand_rot6d_left_modelframe"}
zz = lambda k, a: round(float(np.abs((a - mu[k]) / sd[k]).mean()), 2)  # noqa: E731
res = {"how2sign_test": {k: zz(k, H[k]) for k in H}}
for name, root in (("3dlex_735", HERE / "data/3dlex"), ("wlasl_1000", R / "ver_03_sr_t2p/data/converted")):
    ds = sorted(p for p in root.iterdir() if (p / "meta.json").exists())
    res[name] = {k: zz(k, np.concatenate([np.load(d / f"{F[k]}.npy") for d in ds])) for k in F}
(HERE / "outputs/domain_gap.json").write_text(json.dumps(res, indent=1))
print(res)
