"""Isolated-sign recognizer on joint features (judge/data.py), trained on real WLASL signs only.

    python judge/recognizer.py calib      # 5-fold split BY SIGNER: accuracy on signers never seen in training
    python judge/recognizer.py final      # train on all 1,000 clips -> outputs/judge/recognizer.pt
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D  # noqa: E402

W = 32              # every clip / window is resampled to 32 frames
dev = torch.device("cpu")      # the GPU belongs to the MaDiS run; this net is small
torch.set_num_threads(int(__import__("os").environ.get("JUDGE_THREADS", 8)))


def resample(f, n=W):
    t = np.linspace(0, len(f) - 1, n)
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, len(f) - 1)
    a = (t - i0)[:, None]
    return (f[i0] * (1 - a) + f[i1] * a).astype(np.float32)


def augment(f, rng):
    T = len(f)
    lo = max(2, int(T * rng.uniform(0.0, 0.15)))            # random temporal crop (sign boundaries are fuzzy)
    hi = T - int(T * rng.uniform(0.0, 0.15))
    f = f[min(lo, T - 2):max(hi, min(lo, T - 2) + 2)]
    if rng.random() < 0.5:
        f = D.mirror(f)
    x = f.reshape(len(f), -1, 3)
    th = rng.normal(0, 0.15)                                  # small rotation about the vertical axis + scale
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)
    x = (x @ R.T) * rng.uniform(0.9, 1.1)
    x = x + rng.normal(0, 0.01, x.shape).astype(np.float32)
    return x.reshape(len(f), -1)


class Net(nn.Module):
    def __init__(self, n_cls, d_in=162, d=192, layers=4):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.zeros(1, W, d))
        enc = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.2, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.3), nn.Linear(d, n_cls))

    def forward(self, x):                                     # x (B, W, 162)
        h = self.enc(self.inp(x) + self.pos)
        return self.out(h.mean(1))


EPOCHS = int(__import__("os").environ.get("JUDGE_EPOCHS", 150))


def fit(train, classes, epochs=None, seed=0):
    epochs = epochs or EPOCHS
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    ci = {g: i for i, g in enumerate(classes)}
    net = Net(len(classes)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=epochs * ((len(train) + 63) // 64))
    y_all = np.array([ci[c["gloss"]] for c in train])
    for ep in range(epochs):
        net.train()
        for b in np.array_split(rng.permutation(len(train)), (len(train) + 63) // 64):
            x = torch.tensor(np.stack([resample(augment(train[i]["f"], rng)) for i in b]), device=dev)
            loss = F.cross_entropy(net(x), torch.tensor(y_all[b], device=dev), label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return net.eval()


@torch.no_grad()
def logits(net, feats):
    """feats: list of (T,162). Returns (N, C) max over (original, mirrored) of log-softmax."""
    out = []
    for i in range(0, len(feats), 256):
        x = torch.tensor(np.stack([resample(f) for f in feats[i:i + 256]]), device=dev)
        xm = torch.tensor(np.stack([resample(D.mirror(f)) for f in feats[i:i + 256]]), device=dev)
        out.append(torch.maximum(F.log_softmax(net(x), -1), F.log_softmax(net(xm), -1)).cpu())
    return torch.cat(out).numpy()


def main():
    mode = sys.argv[1]
    clips = D.load_wlasl()
    classes = sorted({c["gloss"] for c in clips})
    D.CACHE.mkdir(parents=True, exist_ok=True)
    if mode == "calib":
        signers = sorted({c["signer"] for c in clips})
        rng = np.random.default_rng(0)
        fold = {s: k for k, s in enumerate(rng.permutation(signers))}
        ranks = []
        for k in range(5):
            te = [c for c in clips if fold[c["signer"]] % 5 == k]
            tr = [c for c in clips if fold[c["signer"]] % 5 != k]
            net = fit(tr, classes)
            lg = logits(net, [c["f"] for c in te])
            ci = [classes.index(c["gloss"]) for c in te]
            r = [(lg[i] > lg[i, ci[i]]).sum() + 1 for i in range(len(te))]
            ranks += r
            print(f"fold {k}: {len(te)} clips, top-1 {np.mean(np.array(r) == 1):.3f}, top-5 {np.mean(np.array(r) <= 5):.3f}",
                  flush=True)
        ranks = np.array(ranks)
        res = {"classes": len(classes), "clips": len(clips), "signers": len(signers),
               "held_out_signer_top1": float(np.mean(ranks == 1)), "top5": float(np.mean(ranks <= 5)),
               "median_rank": float(np.median(ranks)), "chance_top1": 1 / len(classes), "chance_top5": 5 / len(classes)}
        (D.CACHE / "recognizer_calib.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
    else:
        nets = [fit(clips, classes, seed=s) for s in range(3)]   # small ensemble for the judge
        torch.save({"classes": classes, "state": [n.state_dict() for n in nets]}, D.CACHE / "recognizer.pt")
        print("saved", D.CACHE / "recognizer.pt")


if __name__ == "__main__":
    main()
