"""VQ round trip of 3D-LEX with the collar convention aligned to SOKE's How2Sign fittings: the collar rotation is set
to the How2Sign mean collar, and the shoulder takes up the difference so the upper arm keeps its global orientation
(R_sh' = R_col'^T R_col R_sh). Jaw/expression = How2Sign mean. Writes outputs/vq_roundtrip_collarfix.npz
(same layout as vq_roundtrip.npz, 3D-LEX only); evaluate with: vq_eval.py vq_roundtrip_collarfix.npz
    ../ver_03_madis/.venv-madis/bin/python scripts/vq_collarfix.py"""
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, __import__("os").path.dirname(__file__))
import bench_b_madis as bb  # noqa: E402

model, _ = bb.gm.load(bb.DEV)
H = np.concatenate([c["ref"] for c in bb.mc.load_official()[:300]]) * bb.mc.STD + bb.mc.MEAN
out = {}
for d in sorted(p for p in (bb.HERE / "data/3dlex").iterdir() if (p / "meta.json").exists()):
    f = bb.streams_to_madis(*[np.load(d / n) for n in ("body_rot6d.npy", "hand_rot6d.npy", "hand_rot6d_left_modelframe.npy")])
    for col, sh in ((1, 4), (2, 5)):                                  # (l_collar, l_shoulder), (r_collar, r_shoulder)
        c_old = R.from_rotvec(f[:, 3 * col:3 * col + 3])
        c_new = R.from_rotvec(np.repeat(H[:, 3 * col:3 * col + 3].mean(0)[None], len(f), 0))
        s_old = R.from_rotvec(f[:, 3 * sh:3 * sh + 3])
        f[:, 3 * col:3 * col + 3] = c_new.as_rotvec()
        f[:, 3 * sh:3 * sh + 3] = (c_new.inv() * c_old * s_old).as_rotvec()
    f[:, 120:133] = H[:, 120:133].mean(0)
    fn = ((bb.loader_length(f) - bb.mc.MEAN) / bb.mc.STD).astype(np.float32)
    _, dec = bb.tokenize(model, fn)
    out[f"lex_{d.name}_ref"] = fn; out[f"lex_{d.name}_dec"] = dec
zz = np.concatenate([v for k, v in out.items() if k.endswith("_ref")])
print("normalised |z| body / lh / rh:", [round(float(np.abs(zz[:, a:b]).mean()), 2) for a, b in ((0, 30), (30, 75), (75, 120))])
np.savez_compressed(bb.OUT / "vq_roundtrip_collarfix.npz", mean=bb.mc.MEAN, std=bb.mc.STD, **out)
