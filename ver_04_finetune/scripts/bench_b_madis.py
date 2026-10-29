"""Candidates B and C (MaDiS side) on our GPU, measured on a few batches. Nothing is saved except the json/npz below;
the fine-tuned weights only live in memory and are discarded.

1. vq   : can MaDiS's How2Sign VQ-VAEs represent 3D-LEX signs? 3D-LEX streams -> MaDiS 133-d features (inverse of
          ver_03_madis/convert_madis.convert) -> loader length rule (>=40 frames, multiple of 4) -> normalise ->
          encode (body / lhand / rhand VAEs from sft_how2sign.ckpt) -> decode. Also the same round trip for 200
          How2Sign test references (the official dump's feats_ref). Writes outputs/vq_roundtrip.npz; joint errors and
          identity retrieval are computed by scripts/vq_eval.py in the SignSparK venv.
2. step : the SFT step exactly as train.py runs it (MaDiS.train_lm_forward + GPTLosses), forward + backward + AdamW
          step on every non-VAE parameter (full fine-tune, as the released config), for How2Sign batches (official
          test dump: tokens + reference motion) and for word batches (tokenised 3D-LEX signs, word as the caption).
          Precision 32-true (the config) and bf16 autocast.
Fixes ft_benchmark.py's bug (generate_madis.load() returns requires_grad False, so backward() failed).
    ../ver_03_madis/.venv-madis/bin/python scripts/bench_b_madis.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "ver_03_madis"))
import madis_common as mc  # noqa: E402  (MEAN, STD, load_official, resample)
import generate_madis as gm  # noqa: E402  (load; chdir to the madis repo)

DEV = "cuda"
TASK = {"input": ["<Caption_Placeholder>"], "output": ["<Motion_Placeholder>"]}
FLIP = np.diag([1.0, -1.0, -1.0])
TOK = ROOT / "ver_03_madis" / "data" / "How2Sign" / "TOKENS_h2s_csl_phoenix" / "how2sign"
OUT = HERE / "outputs"


def r6(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 /= np.linalg.norm(b2, axis=-1, keepdims=True)
    return np.stack([b1, b2, np.cross(b1, b2)], -2)


def streams_to_madis(body60, rh90, lh90_model):
    """Inverse of convert_madis.convert: SignSparK streams -> de-normalised MaDiS (T, 133); jaw/expression zero."""
    z = np.load(ROOT / "ver_02" / "smplx" / "SMPLX_NEUTRAL.npz", allow_pickle=True)
    T = len(body60)
    aa = lambda R: Rotation.from_matrix(R.reshape(-1, 3, 3)).as_rotvec().reshape(T, -1)  # noqa: E731
    f = np.zeros((T, 133))
    f[:, 0:30] = aa(r6(body60.reshape(T, 10, 6)))
    f[:, 75:120] = aa(r6(rh90.reshape(T, 15, 6))) - z["hands_meanr"]
    f[:, 30:75] = aa(FLIP @ r6(lh90_model.reshape(T, 15, 6)) @ FLIP) - z["hands_meanl"]
    return f


def loader_length(f):
    """sign_dataset.__getitem__: < 40 frames -> resampled to 40; > 400 -> 400; else centre crop to a multiple of 4."""
    if len(f) < 40:
        return mc.resample(f, 40)
    if len(f) > 400:
        return mc.resample(f, 400)
    L = len(f) // 4 * 4
    s = (len(f) - L) // 2
    return f[s:s + L].astype(np.float32)


def split(x):
    return torch.cat([x[..., 0:30], x[..., 120:133]], -1), x[..., 30:75], x[..., 75:120]


@torch.no_grad()
def tokenize(model, fn):
    """fn: normalised (T, 133) -> (T/4, 3) codes, and the decoded (T, 133) normalised features."""
    x = torch.tensor(fn, device=DEV)[None]
    b, l, r = split(x)
    ib, il, ir = model.vae.encode(b)[0], model.hand_vae.encode(l)[0], model.rhand_vae.encode(r)[0]
    db, dl, dr = model.vae.decode(ib), model.hand_vae.decode(il), model.rhand_vae.decode(ir)
    dec = torch.cat([db[..., :30], dl, dr, db[..., 30:]], -1)
    return torch.stack([ib[0], il[0], ir[0]], -1).cpu().numpy(), dec[0].cpu().numpy()


def lex_items(model):
    lex = HERE / "data" / "3dlex"
    items = []
    for d in sorted(p for p in lex.iterdir() if (p / "meta.json").exists()):
        f = streams_to_madis(np.load(d / "body_rot6d.npy"), np.load(d / "hand_rot6d.npy"),
                             np.load(d / "hand_rot6d_left_modelframe.npy"))
        fn = ((loader_length(f) - mc.MEAN) / mc.STD).astype(np.float32)
        tok, dec = tokenize(model, fn)
        items.append({"name": d.name, "text": d.name.replace("_", " "), "tok": tok, "ref": fn, "dec": dec})
    return items


def h2s_items(model):
    items = []
    for c in mc.load_official():
        f = TOK / f"{c['name']}.npy"
        if f.exists():
            t = np.load(f)[0]
            if 4 * len(t) == len(c["ref"]):
                items.append({**c, "tok": t})
    return items


def vq_roundtrip(model, lex, h2s):
    out = {}
    for i, it in enumerate(lex):
        out[f"lex_{it['name']}_ref"] = it["ref"]; out[f"lex_{it['name']}_dec"] = it["dec"]
    rng = np.random.default_rng(0)
    for k in rng.choice(len(h2s), 200, replace=False):
        it = h2s[k]
        tok, dec = tokenize(model, it["ref"])
        same = float((tok == it["tok"]).mean())                         # do we reproduce the released tokens?
        out[f"h2s_{it['name']}_ref"] = it["ref"]; out[f"h2s_{it['name']}_dec"] = dec
        out.setdefault("_h2s_token_agreement", []).append(same)
    agree = float(np.mean(out.pop("_h2s_token_agreement")))
    np.savez_compressed(OUT / "vq_roundtrip.npz", mean=mc.MEAN, std=mc.STD, **out)
    return agree


def batches(items, B, rng, n=7):
    for _ in range(n):
        sel = [items[i] for i in rng.choice(len(items), B, replace=False)]
        L = max(len(c["tok"]) for c in sel)
        tok = torch.zeros(B, L, 3, dtype=torch.long)
        mot = torch.zeros(B, 4 * L, 133)
        for i, c in enumerate(sel):
            tok[i, :len(c["tok"])] = torch.as_tensor(c["tok"])
            mot[i, :len(c["ref"])] = torch.as_tensor(c["ref"])
        yield {"m_tokens": tok.to(DEV), "m_token_len": [len(c["tok"]) for c in sel], "text": [c["text"] for c in sel],
               "tasks": [TASK] * B, "src": ["how2sign"] * B, "name": [c["name"] for c in sel], "motion": mot.to(DEV)}


def step_bench(model, items, label, res):
    vaes = (model.vae, model.hand_vae, model.rhand_vae)
    model.train()
    for v in vaes:
        v.eval()
    for n, p in model.named_parameters():
        p.requires_grad_(not n.startswith(("vae.", "hand_vae.", "rhand_vae.", "metrics", "_metrics")))
    params = [p for p in model.parameters() if p.requires_grad]
    pre = {}
    for n, p in model.named_parameters():
        k = ".".join(n.split(".")[:2]); pre.setdefault(k, [0, 0]); pre[k][int(p.requires_grad)] += p.numel()
    res["param_groups_M_frozen_trainable"] = {k: [round(a / 1e6, 2), round(b / 1e6, 2)] for k, (a, b) in pre.items()}
    print(res["param_groups_M_frozen_trainable"], flush=True)
    res["trainable_params_M"] = sum(p.numel() for p in params) / 1e6
    init = {id(p): p.detach().clone().cpu() for p in params}            # restore after each run (no drift between runs)
    for prec in ("32-true", "bf16-mixed"):
        for B in (8, 16, 32, 64):
            opt = torch.optim.AdamW(params, lr=1e-5, betas=(0.9, 0.99), weight_decay=0.0)
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            key = f"{label}_{prec}_B{B}"
            try:
                ts = []
                for k, b in enumerate(batches(items, B, np.random.default_rng(0))):
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=prec == "bf16-mixed"):
                        out = model.train_lm_forward(b, "train")
                        loss = model._losses["losses_train"].update(out)
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                    torch.cuda.synchronize()
                    if k >= 2:
                        ts.append(time.perf_counter() - t0)
                s = float(np.mean(ts))
                res["runs"][key] = {"sec_per_step": s, "peak_GB": torch.cuda.max_memory_allocated() / 1e9, "loss": float(loss),
                                    "epoch_min_h2s_train_30684": 30684 / B * s / 60, "epoch_min_words_1735": 1735 / B * s / 60}
                print(f"{key}: {s:.3f} s/step, peak {res['runs'][key]['peak_GB']:.1f} GB, loss {float(loss):.3f}", flush=True)
            except torch.OutOfMemoryError:
                res["runs"][key] = "OOM"; print(f"{key}: OOM", flush=True)
            del opt
            opt = None
            with torch.no_grad():
                for p in params:
                    p.copy_(init[id(p)].to(p.device))
            model.zero_grad(set_to_none=True)


def main():
    model, _ = gm.load(DEV)
    res = {"runs": {}}
    t0 = time.time()
    lex = lex_items(model)
    res["lex_tokenize_sec_total"] = time.time() - t0
    res["lex_tokens_median"] = float(np.median([len(i["tok"]) for i in lex]))
    h2s = h2s_items(model)
    res["h2s_tokens_median"] = float(np.median([len(i["tok"]) for i in h2s]))
    res["h2s_released_token_agreement"] = vq_roundtrip(model, lex, h2s)
    print({k: v for k, v in res.items() if k != "runs"}, flush=True)
    step_bench(model, h2s, "h2s", res)
    step_bench(model, lex, "words", res)
    (OUT / "bench_b_madis.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
