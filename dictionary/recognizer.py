"""Isolated-sign recognizer on our SMPL-X dictionary streams (ASL Citizen), the automatic check of the dictionary.

Trained on ASL Citizen's official train split and tested on its test split, whose signers never appear in
training. High unseen-signer accuracy means the WiLoR+NLF extraction keeps each sign's identity (it verifies the
extraction); the trained model then checks every sign of every stitched sentence, and the 3D-LEX / WLASL signs.
Reference: the ASL Citizen paper's own baselines on the same split, from RGB video (I3D) and 2D pose (ST-GCN).

Input per clip: the ver_02 streams (body 60 + right hand 90 + left hand model-frame 90 rot6D, 24 fps; dominance
already canonicalised to the right hand) and their frame differences, resampled to L frames. Model: small
Transformer encoder with a CLS token.

    .venv-sltk/bin/python dictionary/recognizer.py train [--epochs 80]
    .venv-sltk/bin/python dictionary/recognizer.py check        # sentences + 3D-LEX + WLASL
Output: checkpoints/recognizer/aslc_smplx.pt, outputs/dictionary/recognizer_{train,check}.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
DICT = Path(os.environ.get("ASLC_DICT", REPO / "data" / "asl_citizen" / "dict"))   # ASLC_DICT: A/B builds
CKPT = REPO / "checkpoints" / "recognizer" / "aslc_smplx.pt"
OUTD = REPO / "outputs" / "dictionary"
CACHE = DICT / "recognizer_cache.npz"
STREAMS = ("body_rot6d.npy", "hand_rot6d.npy", "hand_rot6d_left_modelframe.npy")
L = 48


def load_streams(d):
    return np.concatenate([np.load(d / k) for k in STREAMS], 1).astype(np.float32)        # (T, 240)


def resample(x, n=L, a=0.0, b=1.0):
    t = np.linspace(a * (len(x) - 1), b * (len(x) - 1), n)
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, len(x) - 1)
    w = (t - i0)[:, None]
    return (1 - w) * x[i0] + w * x[i1]


def to_input(x):
    return np.concatenate([x, np.diff(x, axis=0, prepend=x[:1]) * 4], 1)                  # (L, 480)


class Net(nn.Module):
    def __init__(self, n_cls, d=256, layers=4):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(480, d), nn.LayerNorm(d))
        self.pos = nn.Parameter(torch.zeros(1, L + 1, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, 4, 4 * d, 0.2, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.3), nn.Linear(d, n_cls))

    def forward(self, x):
        h = torch.cat([self.cls.expand(len(x), -1, -1), self.inp(x)], 1) + self.pos
        return self.head(self.enc(h)[:, 0])


def build_cache():
    """All dictionary clips -> raw streams (variable T) + gloss + split; cached once."""
    xs, gl, sp, st = [], [], [], []
    for m in sorted(DICT.glob("clips/**/meta.json")):
        meta = json.loads(m.read_text())
        xs.append(load_streams(m.parent))
        gl.append(meta["gloss"])
        sp.append(meta["split"])
        st.append(m.parent.name)
    lens = np.array([len(x) for x in xs])
    np.savez(CACHE, x=np.concatenate(xs), lens=lens, gloss=np.array(gl), split=np.array(sp), stem=np.array(st))


def load_cache():
    if not CACHE.exists():
        build_cache()
    z = np.load(CACHE)
    off = np.concatenate([[0], np.cumsum(z["lens"])])
    x = z["x"]                                    # npz re-reads the array on every z["x"] access: load it once
    xs = [x[a:b] for a, b in zip(off[:-1], off[1:])]
    return xs, z["gloss"], z["split"], z["stem"]


def batch(xs, idx, train, rng):
    out = []
    for i in idx:
        x = xs[i]
        if train:                                        # temporal crop/speed jitter + small pose noise
            a = rng.uniform(0, 0.1)
            b = rng.uniform(0.9, 1.0)
            x = resample(x, L, a, b) + rng.normal(0, 0.01, (L, x.shape[1])).astype(np.float32)
        else:
            x = resample(x)
        out.append(to_input(x))
    return torch.tensor(np.stack(out), dtype=torch.float32)


@torch.no_grad()
def predict(net, xs, dev, bs=512):
    net.eval()
    P = []
    for i in range(0, len(xs), bs):
        P.append(F.softmax(net(batch(xs, range(i, min(i + bs, len(xs))), False, None).to(dev)), -1).cpu())
    return torch.cat(P)


def topk(P, y, ks=(1, 5, 10)):
    r = (P > P.gather(1, torch.tensor(y)[:, None])).sum(1) + 1
    return {f"top{k}": float((r <= k).float().mean()) for k in ks} | {"n": len(y)}


def train(a):
    xs, gl, sp, _ = load_cache()
    classes = sorted(set(gl[sp == "train"]))
    ci = {g: i for i, g in enumerate(classes)}
    keep = np.array([g in ci for g in gl])
    y = np.array([ci.get(g, -1) for g in gl])
    tr = np.flatnonzero((sp == "train") & keep)
    va = np.flatnonzero((sp == "val") & keep)
    te = np.flatnonzero((sp == "test") & keep)
    print(f"{len(classes)} glosses | train {len(tr)} val {len(va)} test {len(te)} clips", flush=True)
    dev = torch.device(a.device)
    net = Net(len(classes)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=0.05)
    steps = a.epochs * int(np.ceil(len(tr) / a.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(0)
    best, log = -1, []
    for ep in range(a.epochs):
        net.train()
        perm = rng.permutation(tr)
        tot = 0.0
        for i in range(0, len(perm), a.bs):
            idx = perm[i:i + a.bs]
            loss = F.cross_entropy(net(batch(xs, idx, True, rng).to(dev)), torch.tensor(y[idx], device=dev),
                                   label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        v = topk(predict(net, [xs[i] for i in va], dev), y[va])
        log.append({"epoch": ep, "loss": tot / len(tr), **{f"val_{k}": x for k, x in v.items()}})
        print(f"ep {ep:3d} loss {tot / len(tr):.3f} val top1 {v['top1']:.3f} top5 {v['top5']:.3f}", flush=True)
        if v["top1"] > best:
            best = v["top1"]
            CKPT.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state": net.state_dict(), "classes": [str(c) for c in classes]}, CKPT)
    net.load_state_dict(torch.load(CKPT, weights_only=False)["state"])     # own checkpoint (classes: str)
    res = {"classes": len(classes), "val": topk(predict(net, [xs[i] for i in va], dev), y[va]),
           "test_unseen_signers": topk(predict(net, [xs[i] for i in te], dev), y[te]), "log": log}
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / "recognizer_train.json").write_text(json.dumps(res, indent=1))
    print("test (unseen signers):", res["test_unseen_signers"])


def test(a):
    """Score the saved checkpoint on val + test (unseen signers) without retraining."""
    xs, gl, sp, _ = load_cache()
    ck = torch.load(CKPT, weights_only=False)
    ci = {str(g): i for i, g in enumerate(ck["classes"])}
    y = np.array([ci.get(str(g), -1) for g in gl])
    dev = torch.device(a.device)
    net = Net(len(ci)).to(dev)
    net.load_state_dict(ck["state"])
    va, te = (np.flatnonzero((sp == s) & (y >= 0)) for s in ("val", "test"))
    res = {"classes": len(ci), "val": topk(predict(net, [xs[i] for i in va], dev), y[va]),
           "test_unseen_signers": topk(predict(net, [xs[i] for i in te], dev), y[te])}
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / "recognizer_train.json").write_text(json.dumps(res, indent=1))
    print("val:", res["val"], "\ntest (unseen signers):", res["test_unseen_signers"])


def check(a):
    """Rank of the intended gloss for: every sign span of every stitched sentence, 3D-LEX signs, WLASL clips."""
    ck = torch.load(CKPT, weights_only=False)
    classes = ck["classes"]
    ci = {g: i for i, g in enumerate(classes)}
    net = Net(len(classes))
    net.load_state_dict(ck["state"])
    dev = torch.device(a.device)
    net.to(dev)
    sys.path.insert(0, str(REPO / "dictionary"))
    from consensus import LEX_NAME, WORD
    from compare_proto import wm

    def rank_of(x, g):
        p = predict(net, [x], dev)[0]
        cands = [ci[c] for c in classes if WORD(c) == WORD(g)]          # either variant counts
        if not cands:
            return None, None
        r = int((p > p[cands].max()).sum()) + 1
        return r, [classes[i] for i in p.topk(3).indices.tolist()]

    out = {"sentences": {}, "lex": {}, "wlasl": {}, "final": {}}
    for d in sorted((OUTD / "sentences").glob("*/meta.json")):
        meta = json.loads(d.read_text())
        x = load_streams(d.parent)
        out["sentences"][d.parent.name] = [
            {"gloss": g, "rank": rank_of(x[s:e], g)[0], "top3": rank_of(x[s:e], g)[1]} for g, s, e in meta["sign_spans"]]
    for g in sorted({WORD(g) for g in ci}):
        f = DICT / "final" / (g if (DICT / "final" / g).exists() else g + "1")
        if f.exists():
            out["final"][g] = rank_of(load_streams(f), g)[0]
    for g, name in LEX_NAME.items():
        out["lex"][name] = rank_of(load_streams(wm.LEX / name), g)[0]
    for d in sorted((REPO / "data" / "wlasl" / DICT.name / "clips").glob("*/*")):
        out["wlasl"].setdefault(d.parent.name, []).append(rank_of(load_streams(d), d.parent.name)[0])
    sent = [s["rank"] for v in out["sentences"].values() for s in v if s["rank"]]
    out["summary"] = {k: {"n": len(v), "top1": float(np.mean(np.array(v) == 1)), "top5": float(np.mean(np.array(v) <= 5))}
                      for k, v in {"sentence_signs": sent,
                                   "final_dictionary": [r for r in out["final"].values() if r],
                                   "3dlex": [r for r in out["lex"].values() if r],
                                   "wlasl": [r for v in out["wlasl"].values() for r in v if r]}.items()}
    (OUTD / "recognizer_check.json").write_text(json.dumps(out, indent=1))
    for slug, v in out["sentences"].items():
        print(f"{slug:30} " + "  ".join(f"{s['gloss']}:{s['rank']}" for s in v))
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "test", "check", "cache"])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    {"train": train, "test": test, "check": check, "cache": lambda a: build_cache()}[a.mode](a)
