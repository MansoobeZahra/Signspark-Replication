"""Text-only SignSparK (T2P) on a How2Sign test subset, scored with SOKE's DTW-JPE / DTW-PA-JPE (soke_metric.py).

Generation mirrors ver_02/signspark_infer.py (hydra config, CFG wrapper, flow sampler, zero keyframes), batched
over clips. Ground truth = the LMDB clip exactly as SignSparK's loader presents it (truncated to 304 frames).
SMPL-X input as SignSparK's tools/visualize.py builds it (legs/spine/root zero), except the left hand: the LMDB
stores it in true SMPL-X-left convention (verified: left = mirror of right, 0.028 vs 0.427 rad), so GT left is
used as is and only the model's left item (right-hand frame) is flipped.

    .venv-signspark/bin/python eval_t2p.py --name ours                    # our fast-mode setup
    .venv-signspark/bin/python eval_t2p.py --name s50_nocfg --steps 50 --scale -1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import lmdb
import numpy as np
import pickle
import io
import torch
from scipy.spatial.transform import Rotation

import soke_metric as sm

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "SignSparK"
sys.path.insert(0, str(REPO))
from hydra import compose, initialize_config_dir  # noqa: E402

from basic_utils import args_to_dict, create_model_and_flow  # noqa: E402
from classifier_free import ClassifierFreeSampleModel  # noqa: E402

LMDB = ROOT / "data" / "lmdb" / "test" / "How2Sign_reopt_test.lmdb"
OUT = ROOT / "outputs" / "t2p_eval"
T = 304
FLIP = np.diag([1.0, -1.0, -1.0])
PAPER = {"body_pa_jpe": 7.26, "hand_pa_jpe": 2.72, "body_jpe": 6.30, "hand_jpe": 11.43}
dev = torch.device("cuda")


# ---------------------------------------------------------------- data
def load_subset(n, seed=0):
    env = lmdb.open(str(LMDB), readonly=True, lock=False)
    with env.begin() as t:
        ids = pickle.loads(t.get(b"__meta__"))["clip_ids"]
        pick = sorted(np.random.default_rng(seed).choice(len(ids), n, replace=False)) if n < len(ids) else range(len(ids))
        clips = []
        for i in pick:
            z = np.load(io.BytesIO(t.get(ids[i].encode())), allow_pickle=True)
            L = min(len(z["segment"]), T)                           # loader: clips > 304 are truncated
            clips.append({"id": ids[i], "lang": str(z["language"][0]), "text": str(z["translation"][0]), "L": L,
                          "rh": z["right_features"][:L, :90], "lh": z["left_features"][:L, :90],
                          "body": z["body_features"][:L, 66:126]})
    return clips


# ---------------------------------------------------------------- model
def load(stream):
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base=None):
        cfg = compose(config_name="default", overrides=[f"data_v2={stream}", "data_v2/common@common=inference"])
    model, flow = create_model_and_flow(**args_to_dict(cfg, cfg.keys()))
    model.load_state_dict(torch.load(REPO / "checkpoints" / stream / "ema_0.9999_200000.pt", map_location="cpu"),
                          strict=True)
    model.eval().requires_grad_(False).to(dev)
    return model, flow


@torch.no_grad()
def generate(model, flow, texts, lengths, seeds, scale, steps, n_items):
    """Zero-keyframe sampling for a batch of clips; n_items samples per clip, noise seeded per clip.
    scale -1 = the raw model (sample.py's text_guidance_scale -1), else CFG with that scale.
    Returns (B, n_items, T, J)."""
    net = model if scale == -1 else ClassifierFreeSampleModel(model, text_scale=torch.tensor(float(scale), device=dev))
    J, B = model.njoints, len(texts) * n_items
    noise = torch.cat([torch.randn(n_items, J, T, device=dev, generator=torch.Generator(dev).manual_seed(int(s)))
                       for s in seeds])
    L = torch.tensor(np.repeat(lengths, n_items), device=dev)
    zeros = torch.zeros(B, J, T, device=dev)
    obs_mask = torch.zeros(B, J, T, dtype=torch.bool, device=dev)
    kw = {"y": {"mask": (torch.arange(T, device=dev)[None] < L[:, None])[:, None].clone(),
                "lengths": L.cpu(), "text": [t for t in texts for _ in range(n_items)]},
          "obs_x0": zeros, "obs_mask": obs_mask}
    s, _ = flow.decode(net, noise=noise, keyframe_mask=obs_mask, x_embed=zeros, model_kwargs=kw,
                       ode_package="torchdiffeq", ode_stepnum=steps)
    return s.permute(0, 2, 1).float().cpu().numpy().reshape(len(texts), n_items, T, J)


# ---------------------------------------------------------------- SMPL-X input
def r6(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    return np.stack([b1, b2, np.cross(b1, b2)], -2)


def aa(R):
    return Rotation.from_matrix(R.reshape(-1, 3, 3)).as_rotvec().reshape(*R.shape[:-2], 3)


def smplx_in(body60, rh90, lh90, left_in_right_frame):
    n = len(body60)
    body = np.zeros((n, 21, 3))
    body[:, 11:] = aa(r6(body60.reshape(n, 10, 6)))
    Rl = r6(lh90.reshape(n, 15, 6))
    if left_in_right_frame:
        Rl = FLIP @ Rl @ FLIP
    return body, aa(Rl), aa(r6(rh90.reshape(n, 15, 6)))


def summarise(scores):
    m = {k: float(np.mean([s[k] for s in scores])) for k in scores[0] if not k.endswith("_steps")}
    for v in ("jpe", "pa_jpe"):
        m[f"hand_{v}"] = (m[f"lhand_{v}"] + m[f"rhand_{v}"]) / 2
        # readability: per-step mean error in mm (sum / path-length lower bound max(n, m))
        for p in sm.PARTS:
            m[f"{p}_{v}_mm_per_step"] = float(np.mean([s[f"{p}_{v}"] / s[f"{p}_steps"] * 1000 for s in scores]))
    return m


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--tag", default="<American> ", help="prefix; '{lang}' uses the clip's own language field")
    ap.add_argument("--scale", type=float, default=2.5, help="-1 = no CFG (raw model)")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=102)
    ap.add_argument("--length", choices=["real", "304"], default="real")
    ap.add_argument("--no-left-flip", action="store_true", help="feed the model's left item to SMPL-X unflipped")
    ap.add_argument("--hand-mean", choices=["flat", "smplx"], default="flat")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--static", action="store_true", help="predict the subset's mean pose, motionless (no model)")
    ap.add_argument("--gt-baseline", action="store_true", help="also score a DIFFERENT clip's real motion")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    clips = load_subset(a.n)
    models = {s: load(s) for s in ("hand", "body")}
    fk = sm.SoKeJoints(flat_hand_mean=(a.hand_mean == "flat"))
    print(f"loaded {len(clips)} clips + models in {time.perf_counter() - t0:.0f}s", flush=True)

    texts = [(a.tag.format(lang=c["lang"]) if "{lang}" in a.tag else a.tag) + c["text"] for c in clips]
    lengths = [c["L"] if a.length == "real" else T for c in clips]
    seeds = [a.seed + i for i in range(len(clips))]
    t0 = time.perf_counter()
    H, Bd = [], []
    for i in range(0, len(clips), a.batch):
        sl = slice(i, i + a.batch)
        H.append(generate(*models["hand"], texts[sl], lengths[sl], seeds[sl], a.scale, a.steps, 2))
        Bd.append(generate(*models["body"], texts[sl], lengths[sl], seeds[sl], a.scale, a.steps, 1))
    H, Bd = np.concatenate(H), np.concatenate(Bd)
    if a.static:   # mean GT feature over all subset frames, same for every clip and frame (left in right frame)
        allf = lambda k: np.concatenate([c[k] for c in clips])  # noqa: E731
        lh_r = r6(allf("lh").reshape(-1, 15, 6))
        lh_r = (FLIP @ lh_r @ FLIP)[:, :, :2].reshape(-1, 90)     # into the model's right-hand frame
        H[:, 0], H[:, 1] = allf("rh").mean(0), lh_r.mean(0)
        Bd[:, 0] = allf("body").mean(0)
    t_gen = time.perf_counter() - t0
    np.savez_compressed(OUT / f"{a.name}_pred.npz", hand=H, body=Bd, lengths=np.array(lengths),
                        ids=np.array([c["id"] for c in clips]))
    print(f"generation {t_gen:.0f}s", flush=True)

    t0 = time.perf_counter()
    gt_j = [fk(*smplx_in(c["body"], c["rh"], c["lh"], False)) for c in clips]
    scores, base = [], []
    for k, c in enumerate(clips):
        n = lengths[k]
        pred = fk(*smplx_in(Bd[k, 0, :n], H[k, 0, :n], H[k, 1, :n], not a.no_left_flip))
        scores.append({"id": c["id"], **sm.score(pred, gt_j[k])})
        if a.gt_baseline:
            base.append(sm.score(gt_j[(k + 1) % len(clips)], gt_j[k]))
    t_eval = time.perf_counter() - t0

    res = {"args": vars(a), "n": len(clips), "gen_seconds": t_gen, "eval_seconds": t_eval,
           "mean": summarise([{k: v for k, v in s.items() if k != "id"} for s in scores]), "per_clip": scores}
    if base:
        res["baseline_other_clip_gt"] = summarise(base)
    (OUT / f"{a.name}.json").write_text(json.dumps(res, indent=1))

    print(f"\n{a.name}: {len(clips)} clips, gen {t_gen:.0f}s, eval {t_eval:.0f}s")
    print(f"{'':16}{'paper':>8}{'ours':>9}" + (f"{'other GT':>10}" if base else "") + "   ours mm/step")
    for k in ("body_pa_jpe", "hand_pa_jpe", "body_jpe", "hand_jpe"):
        extra = f"{res['baseline_other_clip_gt'][k]:10.2f}" if base else ""
        mm = (res["mean"][f"{k.split('_')[0]}_{k.split('_', 1)[1]}_mm_per_step"] if not k.startswith("hand")
              else (res["mean"][f"lhand_{k.split('_', 1)[1]}_mm_per_step"]
                    + res["mean"][f"rhand_{k.split('_', 1)[1]}_mm_per_step"]) / 2)
        print(f"{k:16}{PAPER[k]:8.2f}{res['mean'][k]:9.2f}{extra}   {mm:6.1f}")


if __name__ == "__main__":
    main()
