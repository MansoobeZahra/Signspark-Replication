"""Candidate A cost on our GPU: SignSparK (frozen) + a residual per-token text branch, measured on a few batches.

The branch: for each hooked block, LayerNorm(x) attends to the per-token XLM-R states of the sentence (the same
states SignSparK mean-pools, unet_model.py:146-148), output projection zero-initialised, added as a residual to
the block's output. At step 0 the model is therefore exactly the released model (checked below).
Hooks are attached with register_forward_hook; no SignSparK file is changed.
Placement: 'mid' = after each of the 8 bottleneck transformer blocks (1024 ch, T=38);
           'mid+lev' = also after each down/up level (7 more, 512-2048 ch, T=38..304).

Batches: real How2Sign TEST clips (the train LMDB is not on disk), padded to 304, text-only training
(obs_mask all False), SignSparK's own loss (flow.training_losses_poses). Each step = text encoding (frozen XLM-R),
forward, backward, AdamW step on the adapter only. Nothing is saved; the weights are discarded at exit.
    ../ver-01/.venv-signspark/bin/python scripts/bench_a_signspark_xattn.py [hand|body]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ver_03_gf_t2p"))
import eval_t2p as ev  # noqa: E402  (load, load_subset, T, dev)

OUT = ROOT / "ver_04_finetune" / "outputs"
dev = ev.dev


class TokenXAttn(nn.Module):
    def __init__(self, dim, ctx_dim=1024, heads=16, head_dim=64):
        super().__init__()
        inner = heads * head_dim
        self.h = heads
        self.norm = nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(ctx_dim)
        self.q = nn.Linear(dim, inner, bias=False)
        self.kv = nn.Linear(ctx_dim, 2 * inner, bias=False)
        self.out = nn.Linear(inner, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, ctx, ctx_mask):
        """x (B, C, T); ctx (B, L, 1024); ctx_mask (B, L) bool (True = real token)."""
        xt = self.norm(x.transpose(1, 2))
        q = self.q(xt).unflatten(-1, (self.h, -1)).transpose(1, 2)
        k, v = self.kv(self.norm_ctx(ctx)).chunk(2, -1)
        k = k.unflatten(-1, (self.h, -1)).transpose(1, 2)
        v = v.unflatten(-1, (self.h, -1)).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=ctx_mask[:, None, None])
        return x + self.out(a.transpose(1, 2).flatten(2)).transpose(1, 2)


class Branch(nn.Module):
    """Holds the adapters and the current text context; installs hooks on the frozen UNet."""

    def __init__(self, model, where):
        super().__init__()
        u = model.unet.unet if hasattr(model.unet, "unet") else model.unet
        targets = [(b, 1024) for b in u.mid_blocks]
        if where == "mid+lev":
            for lev in list(u.downs) + list(u.ups):
                targets.append((lev[2], None))
        self.adapters = nn.ModuleList()
        self.ctx = None
        for mod, dim in targets:
            if dim is None:
                dim = _out_channels(mod)
            ad = TokenXAttn(dim)
            self.adapters.append(ad)
            mod.register_forward_hook(lambda m, i, o, ad=ad: ad(o, *self.ctx) if self.ctx is not None else o)


def _out_channels(mod):
    for p in reversed(list(mod.parameters())):
        if p.dim() == 3:
            return p.shape[0]
    raise ValueError("no conv in module")


@torch.no_grad()
def token_states(model, texts):
    tok = model.text_enc_tokenizer(texts, padding=True, return_tensors="pt").to(dev)
    return model.text_enc_model.transformer(**tok)[0], tok["attention_mask"].bool()


def batch(clips, stream, B, rng):
    sel = [clips[i] for i in rng.choice(len(clips), B, replace=False)]
    J = 90 if stream == "hand" else 60
    x = np.zeros((B, J, ev.T), np.float32)
    for i, c in enumerate(sel):
        a = c["rh"] if stream == "hand" else c["body"]
        x[i, :, :c["L"]] = a.T
        x[i, :, c["L"]:] = a[-1][:, None]                     # the loader pads with the last frame
    L = torch.tensor([c["L"] for c in sel], device=dev)
    kw = {"y": {"mask": (torch.arange(ev.T, device=dev)[None] < L[:, None])[:, None].clone(), "lengths": L.cpu(),
                "text": [f"<{c['lang']}> {c['text']}" for c in sel]},
          "obs_x0": torch.zeros(B, J, ev.T, device=dev), "obs_mask": torch.zeros(B, J, ev.T, dtype=torch.bool, device=dev)}
    return torch.tensor(x, device=dev), kw


def main(stream):
    model, flow = ev.load(stream)
    clips = ev.load_subset(10 ** 9)
    res = {"stream": stream, "base_params_M": sum(p.numel() for p in model.parameters()) / 1e6, "runs": {}}
    for where in ("mid", "mid+lev"):
        br = Branch(model, where).to(dev)
        n = sum(p.numel() for p in br.parameters())
        opt = torch.optim.AdamW(br.parameters(), lr=1e-4, weight_decay=0.01)
        # identity at init: hooked output == unhooked output
        x1, kw = batch(clips, stream, 4, np.random.default_rng(1))
        t = torch.full((4,), 0.5, device=dev)
        with torch.no_grad():
            br.ctx = None
            ref = model(x1, t, **kw)
            br.ctx = token_states(model, kw["y"]["text"])
            same = float((model(x1, t, **kw) - ref).abs().max())
        for B in (16, 32, 64):
            rng = np.random.default_rng(0)
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            try:
                ts, tt = [], []
                for k in range(7):
                    x1, kw = batch(clips, stream, B, rng)
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    br.ctx = token_states(model, kw["y"]["text"])
                    torch.cuda.synchronize(); t1 = time.perf_counter()
                    t = torch.rand(B, device=dev) * (1 - 1e-3) + 1e-3
                    loss = flow.training_losses_poses(model, x1, t, model_kwargs=kw)["loss"].mean()
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                    torch.cuda.synchronize()
                    if k >= 2:
                        ts.append(time.perf_counter() - t0); tt.append(t1 - t0)
                s = float(np.mean(ts))
                r = {"sec_per_step": s, "text_enc_sec": float(np.mean(tt)), "peak_GB": torch.cuda.max_memory_allocated() / 1e9,
                     "loss": float(loss)}
                for name, n_clips in (("h2s_train_29k", 29000), ("h2s_train_29k+3dlex735+wlasl1000", 29000 + 735 + 1000)):
                    r[f"epoch_min_{name}"] = n_clips / B * s / 60
                res["runs"][f"{where}_B{B}"] = r
                print(f"{stream} {where} adapters {n / 1e6:.1f} M (identity at init: max diff {same:.2e}) B={B}: "
                      f"{s:.3f} s/step (text enc {r['text_enc_sec']:.3f}), peak {r['peak_GB']:.1f} GB, "
                      f"{r['epoch_min_h2s_train_29k']:.1f} min/epoch on How2Sign train", flush=True)
            except torch.OutOfMemoryError:
                res["runs"][f"{where}_B{B}"] = "OOM"; print(f"{where} B={B}: OOM", flush=True)
            res[f"{where}_adapter_params_M"] = n / 1e6
            res[f"{where}_identity_at_init_maxdiff"] = same
        br.ctx = None
        for m in model.modules():                             # remove hooks before the next placement
            m._forward_hooks.clear()
        del br, opt
    (OUT / f"bench_a_{stream}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "hand")
