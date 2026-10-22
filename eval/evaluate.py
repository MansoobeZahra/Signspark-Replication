"""Evaluate a SignSparK hand+body checkpoint pair the way the paper's Tab. 3 (T2P) and Tab. 4 (KF2P) do.

Protocol (paper Sec. 4, metric = SOKE's DTW-JPE / DTW-PA-JPE, eval/soke_metric.py):
  * test clips from the released LMDB, truncated to 304 frames as the loader does, generated at their real length;
  * text  = "<{language}> {translation}" exactly as the training loader builds it (specify_lang=True);
  * t2p   = zero keyframes (paper: "zero-keyframe inference, enabled by CFG");
    kf2p  = onset/mid/offset keyframes of every FAST segment (keyframe_selection_mode 3), from the GT clip;
  * the hand model works in the right-hand frame: the GT left hand is flipped into it for keyframes (training
    convention, flip_left_hand=True) and the model's left output is flipped back for scoring;
  * SMPL-X joints as SignSparK's visualize.py builds them (legs/spine/root zero, flat_hand_mean=True).
Controls, same clips, same code path:
  * static     : the dataset's mean pose (from train clips), motionless, at each clip's length;
  * other_text : the same noise but another clip's sentence (t2p only) -> does the text matter?
  * other_seed : the own sentence with a second noise seed  -> how much does the noise matter?

    python eval/evaluate.py --name released_h2s_t2p --dataset How2Sign --mode t2p \
        --hand-ckpt <hand ema.pt> --body-ckpt <body ema.pt> [--n 200] [--scale 2.5] [--steps 10]
"""
import argparse
import io
import json
import pickle
import sys
import time
from pathlib import Path

import lmdb
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "upstream"))
sys.path.insert(0, str(REPO / "eval"))
import soke_metric as sm  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from basic_utils import args_to_dict, create_model_and_flow  # noqa: E402
from classifier_free import ClassifierFreeSampleModel  # noqa: E402
from signspark.pose_datasets_lmdb import PoseDataset_Lmdb, flip_left_hand_features  # noqa: E402

T = 304
FLIP = np.diag([1.0, -1.0, -1.0])
dev = torch.device("cuda")
PAPER = {  # Tab. 3 (GF-T2P, "SignSparK (no KF)") and Tab. 4 (KF2P): body/hand DTW-PA-JPE, DTW-JPE
    ("How2Sign", "t2p"): {"body_pa_jpe": 7.26, "hand_pa_jpe": 2.72, "body_jpe": 6.30, "hand_jpe": 11.43},
    ("CSL-Daily", "t2p"): {"body_pa_jpe": 7.26, "hand_pa_jpe": 2.00, "body_jpe": 6.27, "hand_jpe": 10.63},
    ("PHOENIX-2014T", "t2p"): {"body_pa_jpe": 5.24, "hand_pa_jpe": 1.52, "body_jpe": 4.40, "hand_jpe": 7.10},
    ("How2Sign", "kf2p"): {"body_pa_jpe": 2.11, "hand_pa_jpe": 0.86, "body_jpe": 1.68, "hand_jpe": 2.80},
    ("CSL-Daily", "kf2p"): {"body_pa_jpe": 2.50, "hand_pa_jpe": 0.63, "body_jpe": 2.02, "hand_jpe": 2.71},
    ("PHOENIX-2014T", "kf2p"): {"body_pa_jpe": 2.39, "hand_pa_jpe": 0.71, "body_jpe": 1.92, "hand_jpe": 2.25},
}
METRICS = ("body_pa_jpe", "hand_pa_jpe", "body_jpe", "hand_jpe")


# ---------------------------------------------------------------- data
def lmdb_path(root, split, dataset):
    return Path(root) / split / f"{dataset}_reopt_{split}.lmdb"


def read_clips(path, n, seed=0):
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False)
    with env.begin() as t:
        ids = pickle.loads(t.get(b"__meta__"))["clip_ids"]
        pick = sorted(np.random.default_rng(seed).choice(len(ids), n, replace=False)) if 0 < n < len(ids) \
            else range(len(ids))
        clips = []
        for i in pick:
            z = np.load(io.BytesIO(t.get(ids[i].encode())), allow_pickle=True)
            seg = z["segment"]
            L = min(len(seg), T)
            kf = [start for b in PoseDataset_Lmdb._segment_bounds(seg.tolist()) for start in
                  (b[0], (b[0] + b[1]) // 2, b[1])]
            clips.append({"id": ids[i], "text": f"<{z['language'][0]}> {z['translation'][0]}", "L": L,
                          "kf": sorted({k for k in kf if k < T}),
                          "rh": z["right_features"][:L, :90].astype(np.float32),
                          "lh": z["left_features"][:L, :90].astype(np.float32),
                          "body": z["body_features"][:L, 66:126].astype(np.float32)})
    return clips


def static_pose(root, dataset, n=2000):
    """Mean rot6D feature over up to n train clips (dev if train is missing). Left hand in the right frame."""
    cache = REPO / "outputs" / f"static_{dataset}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["body"], z["rh"], z["lh_rframe"], str(z["source"])
    split = "train" if (lmdb_path(root, "train", dataset) / "data.mdb").exists() else "dev"
    clips = read_clips(lmdb_path(root, split, dataset), n, seed=1)
    cat = lambda k: np.concatenate([c[k] for c in clips])  # noqa: E731
    lh_r = flip_left_hand_features(cat("lh"))
    out = cat("body").mean(0), cat("rh").mean(0), lh_r.mean(0), f"{split}:{len(clips)}"
    cache.parent.mkdir(exist_ok=True)
    np.savez(cache, body=out[0], rh=out[1], lh_rframe=out[2], source=out[3])
    return out


# ---------------------------------------------------------------- model
def load(stream, ckpt):
    with initialize_config_dir(config_dir=str(REPO / "upstream" / "configs"), version_base=None):
        cfg = compose(config_name="default", overrides=[f"data_v2={stream}", "data_v2/common@common=inference"])
    model, flow = create_model_and_flow(**args_to_dict(cfg, cfg.keys()))
    model.load_state_dict(torch.load(ckpt, map_location="cpu", mmap=True, weights_only=True), strict=True)
    return model.eval().requires_grad_(False).to(dev), flow


@torch.no_grad()
def generate(model, flow, x_gt, texts, lengths, keyframes, seeds, scale, steps):
    """x_gt (B, J, T) padded GT features (only read at keyframes); keyframes: list of frame lists ([] = t2p).
    Returns (B, T, J)."""
    net = model if scale == -1 else ClassifierFreeSampleModel(model, text_scale=torch.tensor(float(scale), device=dev))
    B, J, _ = x_gt.shape
    noise = torch.cat([torch.randn(1, J, T, device=dev, generator=torch.Generator(dev).manual_seed(int(s)))
                       for s in seeds])
    L = torch.tensor(lengths, device=dev)
    obs = torch.zeros(B, J, T, dtype=torch.bool, device=dev)
    for b, kf in enumerate(keyframes):
        if kf:
            obs[b, :, kf] = True
    kw = {"y": {"mask": (torch.arange(T, device=dev)[None] < L[:, None])[:, None].clone(),
                "lengths": L.cpu(), "text": list(texts)}, "obs_x0": x_gt, "obs_mask": obs}
    s, _ = flow.decode(net, noise=noise, keyframe_mask=obs, x_embed=x_gt, model_kwargs=kw,
                       ode_package="torchdiffeq", ode_stepnum=steps)
    return s.permute(0, 2, 1).float().cpu().numpy()


def pad(x):
    return np.concatenate([x, np.repeat(x[-1:], T - len(x), 0)]) if len(x) < T else x[:T]


def run_streams(models, clips, idx_text, seed_off, a, kf2p):
    """Generate hand (left, right) and body for every clip. idx_text[k] = which clip's sentence clip k gets."""
    H = np.zeros((len(clips), 2, T, 90), np.float32)
    Bd = np.zeros((len(clips), T, 60), np.float32)
    for i in range(0, len(clips), a.batch):
        cs = clips[i:i + a.batch]
        ks = range(i, i + len(cs))
        texts = [clips[idx_text[k]]["text"] for k in ks]
        lens = [c["L"] for c in cs]
        kfs = [c["kf"] if kf2p else [] for c in cs]
        seeds = [a.seed + seed_off + k for k in ks]
        hx = np.stack([pad(x) for c in cs for x in (flip_left_hand_features(c["lh"]), c["rh"])]).transpose(0, 2, 1)
        h = generate(*models["hand"], torch.tensor(hx, device=dev), [t for t in texts for _ in (0, 1)],
                     [n for n in lens for _ in (0, 1)], [k for k in kfs for _ in (0, 1)],
                     [s * 2 + j for s in seeds for j in (0, 1)], a.scale, a.steps)
        H[i:i + len(cs)] = h.reshape(len(cs), 2, T, 90)
        bx = np.stack([pad(c["body"]) for c in cs]).transpose(0, 2, 1)
        Bd[i:i + len(cs)] = generate(*models["body"], torch.tensor(bx, device=dev), texts, lens, kfs, seeds,
                                     a.scale, a.steps)
    return H, Bd


# ---------------------------------------------------------------- SMPL-X + scoring
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


def score_all(fk, gt_j, clips, H, Bd):
    out = []
    for k, c in enumerate(clips):
        n = c["L"]
        s = sm.score(fk(*smplx_in(Bd[k, :n], H[k, 1, :n], H[k, 0, :n], True)), gt_j[k])
        s["hand_jpe"] = (s["lhand_jpe"] + s["rhand_jpe"]) / 2
        s["hand_pa_jpe"] = (s["lhand_pa_jpe"] + s["rhand_pa_jpe"]) / 2
        out.append(s)
    return out


def mean(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in METRICS}


def feat_dist(A, B, clips):
    """Mean per-frame L2 between two generations in rot6D feature space (hand R+L and body), real frames only."""
    d = []
    for k, c in enumerate(clips):
        n = c["L"]
        d.append(np.linalg.norm(np.concatenate([A[0][k, :, :n].transpose(1, 0, 2).reshape(n, -1), A[1][k, :n]], 1)
                                - np.concatenate([B[0][k, :, :n].transpose(1, 0, 2).reshape(n, -1), B[1][k, :n]], 1),
                                axis=1).mean())
    return float(np.mean(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--hand-ckpt", required=True)
    ap.add_argument("--body-ckpt", required=True)
    ap.add_argument("--dataset", default="How2Sign", choices=["How2Sign", "CSL-Daily", "PHOENIX-2014T"])
    ap.add_argument("--lmdb-root", default=str(REPO / "data" / "lmdb"))
    ap.add_argument("--mode", default="t2p", choices=["t2p", "kf2p"])
    ap.add_argument("--n", type=int, default=200, help="random test subset (rng 0); -1 = whole test set")
    ap.add_argument("--scale", type=float, default=-1, help="CFG text scale; -1 = no guidance")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=102)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--controls", action="store_true", help="also run other_text and other_seed generations")
    a = ap.parse_args()
    out_dir = REPO / "outputs" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    kf2p = a.mode == "kf2p"

    t0 = time.perf_counter()
    clips = read_clips(lmdb_path(a.lmdb_root, "test", a.dataset), a.n)
    models = {"hand": load("hand", a.hand_ckpt), "body": load("body", a.body_ckpt)}
    fk = sm.SoKeJoints(flat_hand_mean=True)
    gt_j = [fk(*smplx_in(c["body"], c["rh"], c["lh"], False)) for c in clips]
    print(f"{len(clips)} clips, models loaded in {time.perf_counter() - t0:.0f}s", flush=True)

    N = len(clips)
    own = list(range(N))
    t0 = time.perf_counter()
    H, Bd = run_streams(models, clips, own, 0, a, kf2p)
    gen_s = time.perf_counter() - t0
    rows = score_all(fk, gt_j, clips, H, Bd)
    res = {"args": vars(a), "n": N, "gen_seconds": gen_s, "paper": PAPER.get((a.dataset, a.mode)),
           "model": mean(rows), "mean_keyframes_per_clip": float(np.mean([len(c["kf"]) for c in clips])) if kf2p else 0}

    sb, srh, slh, src = static_pose(a.lmdb_root, a.dataset)
    Hs = np.broadcast_to(np.stack([slh, srh])[None, :, None], H.shape).copy()
    Bs = np.broadcast_to(sb[None, None], Bd.shape).copy()
    srows = score_all(fk, gt_j, clips, Hs, Bs)
    res["static"] = mean(srows)
    res["static_source"] = src
    res["model_better_than_static_frac"] = {k: float(np.mean([r[k] < s[k] for r, s in zip(rows, srows)]))
                                           for k in METRICS}

    if a.controls:
        other = [(k + N // 2) % N for k in range(N)]       # far index: avoids neighbouring clips of one video
        H2, B2 = run_streams(models, clips, other, 0, a, kf2p)      # same noise, other sentence
        H3, B3 = run_streams(models, clips, own, 10_000, a, kf2p)   # own sentence, other noise
        orows = score_all(fk, gt_j, clips, H2, B2)
        res["other_text"] = mean(orows)
        res["own_vs_other_text"] = {
            k: {"own_better_frac": float(np.mean([r[k] < o[k] for r, o in zip(rows, orows)])),
                "wilcoxon_p_own_better": float(wilcoxon([r[k] for r in rows], [o[k] for o in orows],
                                                        alternative="less").pvalue)} for k in METRICS}
        res["output_change_rot6d"] = {"text_swapped_same_noise": feat_dist((H, Bd), (H2, B2), clips),
                                      "noise_swapped_same_text": feat_dist((H, Bd), (H3, B3), clips)}
        res["output_change_rot6d"]["text_over_noise"] = (res["output_change_rot6d"]["text_swapped_same_noise"]
                                                         / res["output_change_rot6d"]["noise_swapped_same_text"])
    res["per_clip"] = [{"id": c["id"], **{k: r[k] for k in METRICS}} for c, r in zip(clips, rows)]
    (out_dir / f"{a.name}.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(out_dir / f"{a.name}_pred.npz", hand=H, body=Bd, ids=np.array([c["id"] for c in clips]),
                        lengths=np.array([c["L"] for c in clips]))

    print(f"\n{a.name}: {a.dataset} {a.mode}, {N} clips, scale {a.scale}, {a.steps} steps, gen {gen_s:.0f}s")
    hdr = f"{'':14}{'paper':>8}{'model':>8}{'static':>8}{'beats static':>14}"
    print(hdr + (f"{'other text':>12}{'own<other':>11}{'p':>8}" if a.controls else ""))
    for k in METRICS:
        p = res["paper"][k] if res["paper"] else float("nan")
        line = f"{k:14}{p:8.2f}{res['model'][k]:8.2f}{res['static'][k]:8.2f}{res['model_better_than_static_frac'][k]:13.0%}"
        if a.controls:
            ov = res["own_vs_other_text"][k]
            line += f"{res['other_text'][k]:12.2f}{ov['own_better_frac']:10.0%}{ov['wilcoxon_p_own_better']:9.3f}"
        print(line)
    if a.controls:
        print("output change (rot6D):", json.dumps(res["output_change_rot6d"]))


if __name__ == "__main__":
    main()
