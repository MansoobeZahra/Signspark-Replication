"""Official metric code, loaded from the authors' own files (not copies), plus SMPL-X joints as they define them.

* DTW-JPE: MaDiS's released torch implementation (madis/mGPT/metrics/dtw.py, `batched_dtw_jpe`), with MaDiS's
  joint selection (madis/mGPT/utils/human_models.py: upper_body joints, 21-joint hand regressors) exactly as its
  GenerationMetrics.update calls it: body = joints[:, upper_body], align joint 0 (pelvis); hands = regressor @
  vertices, align joint 0 (wrist); hands = (left + right) / 2. MaDiS's README/paper report this number.
* SOKE's original pure-Python `dtw` + `l2_dist_align` (third_party/SOKE/mGPT/metrics/dtw.py, loaded unmodified;
  its only import, `rigid_align`, is taken from SOKE's own human_models.py source) -- used as a cross-check
  and for DTW-PA-JPE on a subset (it is too slow for full test sets).
* DTW-PA-JPE on full sets: ../ver_03_gf_t2p/soke_metric.py, whose fast path was verified equal to SOKE's code
  to 1e-15 (and is cross-checked again here, see `crosscheck`).

Units, as both papers: the DTW accumulated cost in metres (sum over the path), averaged over clips.
"""

import ast
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
MADIS, SOKE = ROOT / "madis", ROOT / "third_party" / "SOKE"
sys.path.insert(0, str(ROOT.parent / "ver_03_gf_t2p"))
import soke_metric as sm  # noqa: E402


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


madis_dtw = _load(MADIS / "mGPT" / "metrics" / "dtw.py", "madis_dtw")      # torch only, no package imports


def _soke_dtw_module():
    """SOKE's dtw.py, unmodified; its `from mGPT.utils.human_models import rigid_align` is satisfied with the two
    functions extracted (by AST, verbatim) from SOKE's human_models.py -- importing that module itself would
    build SMPL-X from SOKE's licence-gated model bundle."""
    src = (SOKE / "mGPT" / "utils" / "human_models.py").read_text()
    tree = ast.parse(src)
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("rigid_transform_3D", "rigid_align")]
    stub = types.ModuleType("mGPT.utils.human_models")
    exec(compile(ast.Module(body=[ast.parse("import numpy as np").body[0], *fns], type_ignores=[]), "soke_hm", "exec"),
         stub.__dict__)
    for m in ("mGPT", "mGPT.utils"):
        sys.modules.setdefault(m, types.ModuleType(m))
    sys.modules["mGPT.utils.human_models"] = stub
    try:
        return _load(SOKE / "mGPT" / "metrics" / "dtw.py", "soke_dtw")
    finally:
        for m in ("mGPT.utils.human_models", "mGPT.utils", "mGPT"):
            sys.modules.pop(m, None)


soke_dtw = _soke_dtw_module()


def dtw_jpe_official(pred, gt, device="cuda"):
    """MaDiS GenerationMetrics.update, verbatim call pattern. pred/gt: SoKeJoints dicts (numpy, metres)."""
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    d = madis_dtw.batched_dtw_jpe([(t(pred["body"]), t(gt["body"]), sm.BODY_IDX, 0),
                                   (t(pred["lhand"]), t(gt["lhand"]), None, 0),
                                   (t(pred["rhand"]), t(gt["rhand"]), None, 0)]).cpu().numpy()
    return {"body_jpe": float(d[0]), "lhand_jpe": float(d[1]), "rhand_jpe": float(d[2]),
            "hand_jpe": float((d[1] + d[2]) / 2)}


def crosscheck(pred, gt):
    """Same pair through all three implementations: SOKE's original (JPE and PA), MaDiS's torch (JPE), our port."""
    from functools import partial
    out = {}
    for part, wanted in (("body", sm.BODY_IDX), ("rhand", None)):
        for name, align in (("jpe", 0), ("pa_jpe", None)):
            out[f"soke_original_{part}_{name}"] = float(
                soke_dtw.dtw(pred[part], gt[part], partial(soke_dtw.l2_dist_align, wanted=wanted, align_idx=align))[0])
    ours = sm.score(pred, gt)
    off = dtw_jpe_official(pred, gt)
    out.update({f"port_{k}": ours[k] for k in ("body_jpe", "body_pa_jpe", "rhand_jpe", "rhand_pa_jpe")})
    out.update({f"madis_official_{k}": off[k] for k in ("body_jpe", "rhand_jpe")})
    return out
