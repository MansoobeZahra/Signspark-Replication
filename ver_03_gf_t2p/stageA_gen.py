"""Stage A generation: noise / seed variants of text-only SignSparK at the OFFICIAL text-only settings
(sample.py README command: no text guidance, 50 Euler steps via torchdiffeq, zero keyframes, real length,
"<American> " + sentence). See REPORT.md "Stage A".

Sentences (59): the 50 test sentences of bt_bestofn (200-subset indices 0-48 + 72), the 3 model_tests render clips
(G1GUMky8kWc_20-2, G095RWKQ39g_13-1, G3FhmHz_7hs_6-10; the last is also test sentence 49) and the 6 demo sentences.
Seed j (0..K-1) is the SAME for every sentence, so seed j means the same initial noise tensor for all sentences
(noise is drawn for the full 304 frames, the length mask decides what is used).

Variants (K seeds each):
  base            raw model, K = 20 (A1 seed study; A4 picks best-of-N from these)
  const           one noise vector per joint channel, repeated over all frames
  lowpass4/16     noise smoothed over time (Gaussian sigma 4 / 16 frames), rescaled to unit std per channel
  tau0.5/0.7/0.9  noise scaled by tau
  textseed        seed = crc32(sentence) + j: each sentence its own, deterministic noise
  textmean0.5/1   noise + alpha * m(e): m = fixed random projection (seed 7) of (e - e_mean), unit RMS
  ctrl_wrong      test sentence i with the TEXT of sentence i+25, own length, same seeds (A4 control), K = 16

    .venv-signspark/bin/python stageA_gen.py base            # -> outputs/stageA/<variant>.npy
"""

import sys
import time
import zlib

import numpy as np
import torch

import bt_common as bc
import gen_common as g
from bt_bestofn import CTRL_SHIFT, DEMO, TAG, sentences

OUTD = bc.ROOT / "outputs" / "stageA"
RENDER_CLIPS = ["G1GUMky8kWc_20-2-rgb_front", "G095RWKQ39g_13-1-rgb_front", "G3FhmHz_7hs_6-10-rgb_front"]
K = {"base": 20, "ctrl_wrong": 16}
NT = 50
BATCH = 16   # entries per forward; small enough to share the GPU with another job


def render_clips():
    import io
    import pickle
    import lmdb
    from eval_t2p import LMDB, T
    env = lmdb.open(str(LMDB), readonly=True, lock=False)
    out = []
    with env.begin() as t:
        for cid in RENDER_CLIPS:
            z = np.load(io.BytesIO(t.get(cid.encode())), allow_pickle=True)
            L = min(len(z["segment"]), T)
            out.append({"id": cid, "text": str(z["translation"][0]), "L": L, "rh": z["right_features"][:L, :90],
                        "lh": z["left_features"][:L, :90], "body": z["body_features"][:L, 66:126]})
    return out


def all_sentences():
    sub, test, demo, fpw = sentences()
    return sub, test, render_clips(), demo


def smooth(noise, sigma):
    r = int(3 * sigma)
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=noise.device) / sigma) ** 2)
    k = (k / k.sum()).view(1, 1, -1)
    B, J, T = noise.shape
    x = torch.nn.functional.conv1d(torch.nn.functional.pad(noise.reshape(B * J, 1, T), (r, r), mode="reflect"), k)
    x = x.reshape(B, J, T)
    return (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True)


def main(variant):
    sub, test, rc, demo = all_sentences()
    k = K.get(variant, 8)
    if variant == "ctrl_wrong":
        S = test
        texts_s = [test[(i + CTRL_SHIFT) % NT]["text"] for i in range(NT)]
    else:
        S = test + rc + demo
        texts_s = [c["text"] for c in S]
    texts = [TAG + t for t in texts_s for _ in range(k)]
    lens = [c["L"] for c in S for _ in range(k)]
    seeds = [j for _ in S for j in range(k)]
    tau = {"tau0.5": 0.5, "tau0.7": 0.7, "tau0.9": 0.9}.get(variant, 1.0)
    if variant == "textseed":
        seeds = [zlib.crc32(t.encode()) % 2 ** 31 + j for t in texts_s for j in range(k)]
    out = {}
    t0 = time.perf_counter()
    for stream, n_items in (("hand", 2), ("body", 1)):
        m, f = g.load(stream)
        tb = g.TextBank(m)
        tb.install()
        noise_fn = None
        if variant == "const":
            noise_fn = lambda n, idx: n[:, :, :1].expand_as(n).contiguous()  # noqa: E731
        elif variant.startswith("lowpass"):
            sig = float(variant[7:])
            noise_fn = lambda n, idx: smooth(n, sig)  # noqa: E731
        elif variant.startswith("textmean"):
            alpha = float(variant[8:])
            e_mean = tb.embed([TAG + c["text"] for c in sub]).mean(0)
            W = torch.randn(m.njoints * g.T, e_mean.numel(), device=g.dev, generator=torch.Generator(g.dev).manual_seed(7))

            def noise_fn(n, idx, W=W, e_mean=e_mean, alpha=alpha, n_items=n_items):
                d = W @ (tb.embed([texts[i] for i in idx]) - e_mean).T                    # (J*T, b)
                d = (d / d.pow(2).mean(0, keepdim=True).sqrt()).T.reshape(-1, m.njoints, g.T)
                return n + alpha * d.repeat_interleave(n_items, 0)
        out[stream] = g.generate(m, f, texts, lens, seeds, n_items, scale=1.0, steps=50, tau=tau, noise_fn=noise_fn,
                                 batch=BATCH)
        del m, f, tb
        torch.cuda.empty_cache()
        print(f"{variant} {stream}: {time.perf_counter() - t0:.0f}s", flush=True)
    feats = np.empty((len(S), k), dtype=object)
    for e in range(len(texts)):
        L = lens[e]
        feats[e // k, e % k] = bc.pred_feats(out["body"][e, 0, :L], out["hand"][e, 1, :L],
                                             out["hand"][e, 0, :L]).astype(np.float16)
    OUTD.mkdir(parents=True, exist_ok=True)
    np.save(OUTD / f"{variant}.npy", feats, allow_pickle=True)
    print(f"saved {variant}: {feats.shape} in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for v in sys.argv[1:]:
        main(v)
