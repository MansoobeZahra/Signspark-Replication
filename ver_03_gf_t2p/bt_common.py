"""SignSparK-BT (back-translation, poses -> English) as a scorer, plus the shared pose-side checks.

BT side: the authors' package (third_party/SignSparK_BT @ 8dd7cdf, HF LionelLow/SignSparK_BT @ 0ced459, How2Sign
model), used unmodified. checkpoints/bt/How2Sign_nopg is the downloaded model directory with `pseudo_gloss_dir`
removed from config.yaml (only needed to train the CTC gloss head; translation does not use it).
Features exactly as the authors' LMDB loader builds them: body_features[:, 66:] (60) | left_features[:, :90]
(TRUE SMPL-X-left, as stored) | right_features[:, :90], truncated to max_sgn_length = 300 frames.
Generated left hands are in the model's right-hand frame and are conjugated back with the authors' own
`flip_left_hand` (the same transform as sample_dumps.load_sample_dump's unflip).

Pose side: 200-way sentence retrieval as in ../ver_03_madis/score_madis.py (copied: body 12 joints pelvis-relative
+ both hands wrist-relative, resampled to 64 frames, DTW / 128).
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "third_party" / "SignSparK_BT"))
from signspark_bt.cli import _normalize  # noqa: E402
from signspark_bt.data.sample_dumps import flip_left_hand  # noqa: E402
from signspark_bt.external_metrics import sacrebleu  # noqa: E402
from signspark_bt.helpers import load_config  # noqa: E402
from signspark_bt.inference import back_translate, make_back_translation_model  # noqa: E402
from signspark_bt.metrics import score_translations  # noqa: E402

import soke_metric as sm  # noqa: E402

BT_DIR = ROOT / "checkpoints" / "bt" / "How2Sign_nopg"
MAX_SGN = 300


class BT:
    def __init__(self):
        self.cfg = load_config(BT_DIR / "config.yaml")["data"]
        self.model = make_back_translation_model(BT_DIR)

    def ref(self, sentence):
        return _normalize(sentence, self.cfg)

    def translate(self, feats, batch_size=64):
        return back_translate(self.model, [torch.as_tensor(f[:MAX_SGN], dtype=torch.float32) for f in feats],
                              batch_size=batch_size)

    def corpus(self, refs, hyps):
        s = score_translations(refs, hyps, self.cfg.get("version", ""), self.cfg.get("level", "word"))
        return {**{f"bleu{n}": s["bleu"][f"bleu{n}"] for n in range(1, 5)}, "chrf": s["chrf"], "rouge": s["rouge"]}


def sent_chrf(ref, hyp):
    """Sentence-level chrF (x100), the per-candidate selection score."""
    return sacrebleu.corpus_chrf([hyp], [ref]).score * 100


def gt_feats(c):
    """LMDB clip (eval_t2p.load_subset dict, which keeps body_features[:, 66:126]) -> (L, 240)."""
    return np.concatenate([c["body"], c["lh"], c["rh"]], 1).astype(np.float32)


def pred_feats(body60, left_rframe, right):
    """Generated streams (left hand in the model's right-hand frame) -> (L, 240), left back to true SMPL-X-left."""
    return np.concatenate([body60, flip_left_hand(left_rframe), right], 1).astype(np.float32)


# ---------------------------------------------------------------- pose side
def retrieval_feats(J):
    """(T, ...) SoKeJoints dict -> (64, 54, 3): body 12 joints (pelvis-relative) + both hands wrist-relative."""
    b = J["body"][:, sm.BODY_IDX] - J["body"][:, :1]
    lh, rh = J["lhand"] - J["lhand"][:, :1], J["rhand"] - J["rhand"][:, :1]
    x = np.concatenate([b, lh, rh], 1)
    t = np.linspace(0, len(x) - 1, 64)
    return np.stack([np.stack([np.interp(t, np.arange(len(x)), x[:, k, c]) for c in range(3)], -1)
                     for k in range(x.shape[1])], 1)


def dtw_batch(C):
    """sm.dtw_fast over a batch of cost matrices (B, r, c), same recursion, on the GPU."""
    B, r, c = C.shape
    D = torch.full((B, r + 1, c + 1), float("inf"), dtype=C.dtype, device=C.device)
    D[:, 0, 0] = 0.0
    for d in range(r + c - 1):
        i = torch.arange(max(0, d - c + 1), min(r, d + 1), device=C.device)
        j = d - i
        D[:, i + 1, j + 1] = C[:, i, j] + torch.minimum(torch.minimum(D[:, i, j], D[:, i, j + 1]), D[:, i + 1, j])
    return D[:, r, c]


def dist_matrix(preds, gts):
    """(P, 64, 54, 3) x (G, ...) -> (P, G) length-normalised DTW distances (as ver_03_madis retrieval)."""
    G = torch.tensor(np.stack(gts), device="cuda")
    D = np.zeros((len(preds), len(gts)))
    for i, p in enumerate(preds):
        C = (torch.tensor(p, device="cuda")[None, :, None] - G[:, None]).norm(dim=-1).mean(-1)
        D[i] = (dtw_batch(C) / 128).cpu().numpy()
    return D


def retrieval(D, own):
    """D (P, G); own[i] = column of prediction i's real clip. Rank of the own clip, 1 = best."""
    rank = np.array([(D[i] < D[i, own[i]]).sum() + 1 for i in range(len(D))])
    G = D.shape[1]
    return {"top1": float((rank == 1).mean()), "top5": float((rank <= 5).mean()),
            "median_rank": float(np.median(rank)), "mean_rank": float(rank.mean()),
            "chance_top1": 1 / G, "chance_top5": 5 / G, "chance_median_rank": (G + 1) / 2}


def motion_energy(feats):
    """Mean absolute frame-to-frame change of the 240-d rot6D features (0 = standing still)."""
    return float(np.abs(np.diff(feats, axis=0)).mean())
