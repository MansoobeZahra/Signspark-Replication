"""D1: how much sentence information reaches SignSparK's UNet?

The UNet sees text only through one vector: c = unet.time_mlp(embed_timestep(t) + embed_text(mclip(text))).
This script measures, for a released (or retrained) checkpoint, on the How2Sign test sentences:
  - pairwise cosine of the pooled M-CLIP sentence embeddings (raw and with the "<American> " tag);
  - the spread of the final conditioning vector c across sentences (at fixed t), compared with
    its spread across timesteps (fixed sentence) and the text-vs-null difference.
No pose data and no UNet forward pass is needed.

    python diagnostics/d1_text_conditioning.py --ckpt <ema.pt> --lmdb <How2Sign_reopt_test.lmdb> --out outputs/d1_<name>.json
"""
import argparse
import io
import json
import pickle
import sys
from pathlib import Path

import lmdb
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "upstream"))
from signspark.unet_model import TimestepEmbedder  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"


def read_sentences(path, limit):
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        ids = pickle.loads(txn.get(b"__meta__"))["clip_ids"][:limit]
        out = []
        for k in ids:
            z = np.load(io.BytesIO(txn.get(k.encode())), allow_pickle=True)
            out.append((str(z["language"][0]), str(z["translation"][0])))
    return out


@torch.no_grad()
def mclip_encode(texts, bs=128):
    from multilingual_clip import pt_multilingual_clip
    import transformers
    ver = "M-CLIP/XLM-Roberta-Large-Vit-B-16Plus"
    model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(ver).to(dev).eval()
    tok = transformers.AutoTokenizer.from_pretrained(ver)
    embs = []
    for i in range(0, len(texts), bs):
        t = tok(texts[i:i + bs], padding=True, return_tensors="pt").to(dev)
        h = model.transformer(**t)[0]
        h = (h * t["attention_mask"].unsqueeze(-1)).sum(1) / t["attention_mask"].sum(1, keepdim=True)
        embs.append(model.LinearTransformation(h).float())
    return torch.cat(embs)


def cos_stats(e):
    e = torch.nn.functional.normalize(e, dim=-1)
    s = e @ e.T
    iu = torch.triu_indices(len(e), len(e), 1)
    v = s[iu[0], iu[1]]
    return {"mean": v.mean().item(), "p5": v.quantile(0.05).item(), "p95": v.quantile(0.95).item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lmdb", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pairs = read_sentences(a.lmdb, a.n)
    plain = [s for _, s in pairs]
    tagged = [f"<{l}> {s}" for l, s in pairs]
    e_plain, e_tag = mclip_encode(plain), mclip_encode(tagged)

    sd = torch.load(a.ckpt, map_location="cpu", mmap=True, weights_only=True)
    lin = torch.nn.Linear(640, 512).to(dev)
    lin.load_state_dict({"weight": sd["embed_text.weight"], "bias": sd["embed_text.bias"]})
    te = TimestepEmbedder(512).to(dev)
    te.load_state_dict({k.split("embed_timestep.")[1]: v for k, v in sd.items() if k.startswith("embed_timestep.")})
    mlp = torch.nn.Sequential(torch.nn.Linear(512, 2048), torch.nn.SiLU(), torch.nn.Linear(2048, 2048),
                              torch.nn.SiLU(), torch.nn.Linear(2048, 512)).to(dev)
    mlp.load_state_dict({k.split("unet.time_mlp.")[1]: v for k, v in sd.items() if k.startswith("unet.time_mlp.")})

    with torch.no_grad():
        ts = torch.linspace(0.001, 1.0, 21, device=dev)
        temb = te(ts)                                    # (21, 512)
        txt = lin(e_tag)                                 # (N, 512)
        null = lin(torch.zeros(1, 640, device=dev))      # (1, 512) = bias (CFG null)
        res = {}
        for name, ti in [("t=0.0", 0), ("t=0.5", 10), ("t=1.0", 20)]:
            c_txt = mlp(temb[ti] + txt)                  # (N, 512): final conditioning, varying sentence
            c_null = mlp(temb[ti] + null)
            c_mean = c_txt.mean(0, keepdim=True)
            res[name] = {
                "sentence_spread": (c_txt - c_mean).norm(dim=-1).mean().item(),
                "text_vs_null": (c_txt - c_null).norm(dim=-1).mean().item(),
                "cond_norm": c_txt.norm(dim=-1).mean().item(),
            }
        c_time = mlp(temb + txt[:1])                     # one sentence, varying t
        res["time_spread_fixed_sentence"] = (c_time - c_time.mean(0, keepdim=True)).norm(dim=-1).mean().item()
        res["pre_mlp"] = {
            "text_sentence_spread": (txt - txt.mean(0)).norm(dim=-1).mean().item(),
            "text_vs_null": (txt - null).norm(dim=-1).mean().item(),
            "time_spread": (temb - temb.mean(0)).norm(dim=-1).mean().item(),
        }
    out = {
        "ckpt": a.ckpt, "n_sentences": len(pairs),
        "mclip_cosine_plain": cos_stats(e_plain), "mclip_cosine_tagged": cos_stats(e_tag),
        "tag_shift_vs_sentence_spread": ((e_tag - e_plain).norm(dim=-1).mean() /
                                         (e_plain - e_plain.mean(0)).norm(dim=-1).mean()).item(),
        "conditioning": res,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
