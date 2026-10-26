"""Sign-level judge, data side: every motion source -> the same per-frame joint features.

Sources (all on disk, nothing generated here):
  wlasl      : SignAvatars WLASL SMPL-X fits, 1,000 isolated-sign clips, 124 glosses ("a".."available"), 53 signers,
               converted to SignSparK streams by ../ver_03_sr_t2p/convert_signavatars.py (24 fps).
  h2s_gt     : How2Sign test, SignSparK's re-optimised fits (LMDB), 2,183 clips, native fps (24/30/50).
  signspark  : official SignSparK text-only output (sample.py, 50 Euler steps, no CFG) for those 2,183 clips
               (model_tests/signspark/ckpt/*/..._t2p.npy). The model's left hand is in the right-hand frame.
  soke_gt / madis : MaDiS official test.py dump on SOKE's How2Sign fittings (24 fps), 2,308 clips: feats_ref (real)
               and feats_rst (MaDiS output), converted by ../ver_03_madis/convert_madis.py (verified to 0.0003 mm).
Every motion is reduced to (body60 rot6D, right90, left90 TRUE SMPL-X-left), then SMPL-X joints exactly as
eval/soke_metric.SoKeJoints computes them (SOKE betas, legs/spine zero, flat_hand_mean=True), then features:
  12 upper-body joints relative to the neck, 2 x 21 hand joints relative to their wrist, all divided by the
  shoulder width (per clip) -> (T, 162). Handedness: features of the mirrored body are available (mirror()).
"""
import glob
import io
import json
import pickle
import re
import sys
from pathlib import Path

import lmdb
import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
sys.path.insert(0, str(REPO / "eval"))
import soke_metric as sm  # noqa: E402
from evaluate import smplx_in  # noqa: E402

WLASL_DIR = ROOT / "ver_03_sr_t2p" / "data" / "converted"
WLASL_JSON = ROOT / "ver_02" / "data" / "WLASL_v0.3.json"
H2S_LMDB = REPO / "data" / "lmdb" / "test" / "How2Sign_reopt_test.lmdb"
SS_DUMP = ROOT / "model_tests" / "signspark" / "ckpt"
SS_T2P = {"hand": SS_DUMP / "hand" / "debug0_eulerstepsize50_can1_anchor0_samples_20261019-10:17:12" / "seed102_clampstep0_t2p.npy",
          "body": SS_DUMP / "body" / "debug0_eulerstepsize50_can1_anchor0_samples_20261019-10:33:36" / "seed102_clampstep0_t2p.npy"}
MADIS_PRED = ROOT / "model_tests" / "madis" / "results" / "madis" / "madis_how2sign" / "test_rank_0"
CACHE = REPO / "outputs" / "judge"
FLIP = np.diag([1.0, -1.0, -1.0])
BI = sm.BODY_IDX            # neck, L/R shoulder, L/R elbow, L/R wrist, ears, eyes, nose (SMPL-X indices)

_FK = None


def fk():
    global _FK
    if _FK is None:
        _FK = sm.SoKeJoints(flat_hand_mean=True, device="cpu")   # CPU: the GPU belongs to the MaDiS run
    return _FK


def r6_rows(R):
    return R[..., :2, :].reshape(*R.shape[:-3], -1)


def left_true_from_model(lh90):
    """left hand in the model's right-hand frame -> true SMPL-X left (rot6D rows)."""
    T = len(lh90)
    R = sm_r6(lh90.reshape(T, 15, 6))
    return (FLIP @ R @ FLIP)[:, :, :2].reshape(T, 90)


def sm_r6(d6):
    from evaluate import r6
    return r6(d6)


def features(body60, rh90, lh90_true):
    """-> (T, 162) float32."""
    b, l, r = smplx_in(np.asarray(body60), np.asarray(rh90), np.asarray(lh90_true), False)
    J = fk()(b, l, r)
    body = J["body"][:, BI]                                   # (T, 12, 3)
    neck = body[:, :1]
    sw = np.linalg.norm(body[:, 1] - body[:, 2], axis=-1).mean() + 1e-6
    f = np.concatenate([(body - neck).reshape(len(body), -1),
                        (J["lhand"] - J["lhand"][:, :1]).reshape(len(body), -1),
                        (J["rhand"] - J["rhand"][:, :1]).reshape(len(body), -1)], 1) / sw
    return f.astype(np.float32)


# body joint order in BI: 0 neck, 1 L sh, 2 R sh, 3 L elb, 4 R elb, 5 L wr, 6 R wr, 7 R ear?, ... mirror pairs:
_BODY_SWAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 11]          # BI = [12,16,17,18,19,20,21,59,58,57,56,55]


def mirror(f):
    """Features of the left-right mirrored signer (x -> -x, swap sides, swap hands)."""
    T = len(f)
    body = f[:, :36].reshape(T, 12, 3)[:, _BODY_SWAP].copy()
    lh, rh = f[:, 36:99].reshape(T, 21, 3).copy(), f[:, 99:].reshape(T, 21, 3).copy()
    for a in (body, lh, rh):
        a[..., 0] *= -1
    return np.concatenate([body.reshape(T, -1), rh.reshape(T, -1), lh.reshape(T, -1)], 1)


def words(text):
    return set(re.findall(r"[a-z']+", text.lower()))


# ---------------------------------------------------------------- loaders (cached as npz of object arrays)
def _cache(name, build):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"feat_{name}.pkl"
    if p.exists():
        return pickle.load(open(p, "rb"))
    out = build()
    pickle.dump(out, open(p, "wb"))
    return out


def load_wlasl():
    def build():
        W = json.load(open(WLASL_JSON))
        info = {i["video_id"]: (e["gloss"], i["signer_id"]) for e in W for i in e["instances"]}
        out = []
        for d in sorted(glob.glob(str(WLASL_DIR / "*"))):
            vid = Path(d).name
            body = np.load(f"{d}/body_rot6d.npy")
            rh = np.load(f"{d}/hand_rot6d.npy")
            lh = left_true_from_model(np.load(f"{d}/hand_rot6d_left_modelframe.npy"))
            out.append({"id": vid, "gloss": info[vid][0], "signer": info[vid][1], "f": features(body, rh, lh)})
        return out
    return _cache("wlasl", build)


def load_h2s_gt():
    def build():
        env = lmdb.open(str(H2S_LMDB), readonly=True, lock=False, readahead=False)
        out = []
        with env.begin() as t:
            for cid in pickle.loads(t.get(b"__meta__"))["clip_ids"]:
                z = np.load(io.BytesIO(t.get(cid.encode())), allow_pickle=True)
                L = min(len(z["segment"]), 304)
                out.append({"id": cid, "text": str(z["translation"][0]),
                            "f": features(z["body_features"][:L, 66:126], z["right_features"][:L, :90],
                                          z["left_features"][:L, :90])})
        return out
    return _cache("h2s_gt", build)


def load_signspark_t2p():
    def build():
        h = np.load(SS_T2P["hand"], allow_pickle=True).item()
        b = np.load(SS_T2P["body"], allow_pickle=True).item()
        hp, bp = h["pred_poses"][0], b["pred_poses"][0]          # (2B, 90, 304) L0,R0,L1,R1 / (B, 60, 304)
        L, names, texts = b["lengths"][0], b["video_names"][0], b["text"][0]
        assert list(h["video_names"][0][::2]) == list(names)
        out = []
        for k in range(len(bp)):
            n = int(L[k])
            out.append({"id": str(names[k]), "text": str(texts[k]).split("> ", 1)[-1],
                        "f": features(bp[k, :, :n].T, hp[2 * k + 1, :, :n].T, left_true_from_model(hp[2 * k, :, :n].T))})
        return out
    return _cache("signspark_t2p", build)


def load_madis():
    """-> (soke_gt, madis) lists, same clips."""
    sys.path.insert(0, str(ROOT / "ver_03_madis"))
    import madis_common as mc  # noqa: E402  (MEAN/STD, convert)

    def build():
        gt, pr = [], []
        for f in sorted(MADIS_PRED.glob("*.pkl")):
            d = pickle.load(open(f, "rb"))
            for lst, key in ((gt, "feats_ref"), (pr, "feats_rst")):
                body, rh, lh_r = mc.convert(np.asarray(d[key], np.float64) * mc.STD + mc.MEAN)
                lst.append({"id": f.stem, "text": str(d["text"]), "f": features(body, rh, left_true_from_model(lh_r))})
        return {"soke_gt": gt, "madis": pr}
    out = _cache("madis", build)
    return out["soke_gt"], out["madis"]
