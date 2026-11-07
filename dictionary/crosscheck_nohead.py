"""Re-score consensus.py's cross-check with the face points left out of the word metric.

ver_04's retrieval features use 12 body points of which 5 are face landmarks (SMPL-X 55-59: nose, eyes, ears), so
head pitch weighs heavily; 3D-LEX's avatar has its neck pitched ~22 deg forward. To judge the manual sign only,
this scores the same queries (3D-LEX signs, WLASL clips built by the same pipeline) against the same bank
(the dictionary's consensus signs) with body points = neck, shoulders, elbows, wrists only.

    CUDA_VISIBLE_DEVICES= .venv-sltk/bin/python dictionary/crosscheck_nohead.py dict_noarm dict dict_v3
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import consensus as cs  # noqa: E402

cs.wm.sm.BODY_IDX[:] = [12, 16, 17, 18, 19, 20, 21]
REPO = cs.REPO

for name in sys.argv[1:]:
    D = REPO / "data" / "asl_citizen" / name
    gl = sorted(p.name for p in (D / "consensus").iterdir())
    bank = [cs.feat_pair(cs.load(D / "consensus" / g))[0] for g in gl]
    lx = [cs.closed_rank(*cs.feat_pair(cs.load(cs.wm.LEX / n)), g, gl, bank)[0] for g, n in cs.LEX_NAME.items()]
    wl = []
    for gd in sorted((REPO / "data" / "wlasl" / name / "clips").iterdir()):
        own = next((g for g in gl if cs.WORD(g) == gd.name), None)
        if own:
            wl += [cs.closed_rank(*cs.feat_pair(cs.load(c)), own, gl, bank)[0] for c in sorted(gd.iterdir())]
    lx, wl = np.array(lx), np.array(wl)
    print(f"{name:12} (no face points) 3D-LEX top1 {np.mean(lx == 1):.2f} top3 {np.mean(lx <= 3):.2f} mean {lx.mean():.2f}"
          f" | WLASL n={len(wl)} top1 {np.mean(wl == 1):.2f} top3 {np.mean(wl <= 3):.2f} mean {wl.mean():.2f}", flush=True)
