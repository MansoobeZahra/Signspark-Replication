"""Text-only SignSparK generation with generation-time hooks (no retraining), for steps 2-3.

Same model, checkpoints, flow sampler and zero-keyframe setup as eval_t2p.generate (which mirrors
ver_02/signspark_infer.py). Hooks:
  - TextBank: M-CLIP embeddings computed once per string and cached (the stock model re-encodes the text at every
    ODE step); synthetic keys can hold any 640-d vector, e.g. the mean sentence embedding or e_mean + lam*(e - e_mean).
  - Guided: v = v_neg + s * (v_text - v_neg). neg = None -> the model's own null condition (zero text vector),
    i.e. exactly ClassifierFreeSampleModel; neg = a TextBank key -> negative-prompt guidance.
  - Composite: per-chunk text conditioning over time windows. At every ODE step the model runs once per chunk text;
    the velocities are blended with soft time masks (MultiDiffusion-style), then guided as above.
  - temperature: initial noise scaled by tau.
  - noise_fn: any other change to the initial noise (structured, text-derived).
"""

import numpy as np
import torch

from eval_t2p import T, dev, load  # noqa: F401  (load re-exported)


class TextBank:
    def __init__(self, model):
        self.model, self.enc, self.bank = model, model.encode_text, {}

    def embed(self, texts):
        new = sorted({t for t in texts if t not in self.bank})
        for i in range(0, len(new), 64):
            e = self.enc(new[i:i + 64])
            self.bank.update({t: e[k] for k, t in enumerate(new[i:i + 64])})
        return torch.stack([self.bank[t] for t in texts])

    def put(self, key, vec):
        self.bank[key] = vec.to(dev).float()

    def install(self):
        self.model.encode_text = lambda raw: torch.stack([self.bank[t] for t in raw]) \
            if all(t in self.bank for t in raw) else self.embed(raw)


class Guided(torch.nn.Module):
    def __init__(self, model, scale, neg=None):
        super().__init__()
        self.model, self.scale, self.neg = model, float(scale), neg
        self.njoints, self.nfeats = model.njoints, model.nfeats

    def neg_out(self, x, t, y, obs_x0, obs_mask):
        yn = dict(y)
        if self.neg is None:
            yn["uncond"] = True
        else:
            yn["text"] = [self.neg] * len(y["text"])
        return self.model(x, t, yn, obs_x0, obs_mask)

    def cond_out(self, x, t, y, obs_x0, obs_mask):
        return self.model(x, t, y, obs_x0, obs_mask)

    def forward(self, x, t, y=None, obs_x0=None, obs_mask=None, **kw):
        c = self.cond_out(x, t, y, obs_x0, obs_mask)
        if self.scale == 1.0:
            return c
        n = self.neg_out(x, t, y, obs_x0, obs_mask)
        return n + self.scale * (c - n)


class Composite(Guided):
    """y['chunks']: per sample a list of K chunk texts; y['wmask']: (B, K, T) soft time weights (sum 1 over K).
    Model outputs are (B, J, T), so chunk k's weight is broadcast as (B, 1, T)."""

    def cond_out(self, x, t, y, obs_x0, obs_mask):
        K = y["wmask"].shape[1]
        out = 0
        for k in range(K):
            yk = dict(y)
            yk["text"] = [ch[k] for ch in y["chunks"]]
            out = out + y["wmask"][:, k:k + 1] * self.model(x, t, yk, obs_x0, obs_mask)
        return out


def chunk_windows(text, L, K=3, ramp=9):
    """Split the sentence into K word chunks; chunk k owns a time span of the clip proportional to its word count,
    cross-faded over `ramp` frames (box filter). Sentences with < K words repeat the last chunk with zero weight.
    Returns (K chunk texts, (K, T) weights summing to 1 on every frame)."""
    words = text.split()
    k_eff = min(K, len(words))
    parts = np.array_split(np.arange(len(words)), k_eff)
    chunks = [" ".join(words[p[0]:p[-1] + 1]) for p in parts]
    edges = np.concatenate([[0], np.cumsum([len(p) for p in parts])]) / len(words) * L
    owner = np.minimum(np.searchsorted(edges[1:], np.arange(T) + 0.5, side="right"), k_eff - 1)
    w = np.zeros((K, T))
    w[owner, np.arange(T)] = 1.0
    box = np.ones(ramp) / ramp
    w = np.stack([np.convolve(np.pad(r, ramp // 2, mode="edge"), box, "valid") for r in w])
    w = w / w.sum(0, keepdims=True)
    return chunks + [chunks[-1]] * (K - k_eff), w


@torch.no_grad()
def generate(model, flow, texts, lengths, seeds, n_items, scale=2.5, steps=10, neg=None, tau=1.0,
             chunks=None, wmask=None, batch=64, noise_fn=None):
    """One entry per (text, length, seed); n_items samples each (noise seeded per entry, as eval_t2p).
    noise_fn(noise (B*n_items, J, T), entry indices) -> noise: structured / text-derived noise hook, applied
    after the tau scaling. Returns (E, n_items, T, J) float32."""
    J = model.njoints
    if chunks is not None:
        net = Composite(model, scale, neg)
    else:
        net = Guided(model, scale, neg)
    outs = []
    for i in range(0, len(texts), batch):
        tx, Ls, sd = texts[i:i + batch], lengths[i:i + batch], seeds[i:i + batch]
        B = len(tx) * n_items
        noise = tau * torch.cat([torch.randn(n_items, J, T, device=dev, generator=torch.Generator(dev).manual_seed(int(s)))
                                 for s in sd])
        if noise_fn is not None:
            noise = noise_fn(noise, list(range(i, i + len(tx))))
        L = torch.tensor(np.repeat(Ls, n_items), device=dev)
        zeros = torch.zeros(B, J, T, device=dev)
        obs = torch.zeros(B, J, T, dtype=torch.bool, device=dev)
        y = {"mask": (torch.arange(T, device=dev)[None] < L[:, None])[:, None].clone(), "lengths": L.cpu(),
             "text": [t for t in tx for _ in range(n_items)]}
        if chunks is not None:
            y["chunks"] = [c for c in chunks[i:i + batch] for _ in range(n_items)]
            y["wmask"] = torch.tensor(np.repeat(np.stack(wmask[i:i + batch]), n_items, 0), dtype=torch.float32,
                                      device=dev)
        s, _ = flow.decode(net, noise=noise, keyframe_mask=obs, x_embed=zeros,
                           model_kwargs={"y": y, "obs_x0": zeros, "obs_mask": obs},
                           ode_package="torchdiffeq", ode_stepnum=steps)
        outs.append(s.permute(0, 2, 1).float().cpu().numpy().reshape(len(tx), n_items, T, J))
    return np.concatenate(outs)
