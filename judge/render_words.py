"""Visual check of the spotting result: for a word, render (CPU, no GPU)
  row 1  the real isolated WLASL sign (the clip the recognizer is most sure of),
  row 2  real How2Sign signing of a sentence containing the word: the window where the judge finds the sign,
  row 3  SignSparK text-only for the same sentence: its best window for that sign,
  row 4  MaDiS for the same sentence: its best window for that sign,
each as 5 frames across the window (front view). Sentences are chosen on the REAL signing (highest score), so the
model rows show what the model does where a real signer signs the word.
    python judge/render_words.py all after again add     # -> outputs/judge/words/<word>.png
"""
import io
import pickle
import sys
from pathlib import Path

import imageio
import lmdb
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D  # noqa: E402
import recognizer as R  # noqa: E402
import spot as SP  # noqa: E402

sys.path.insert(0, str(D.ROOT / "model_tests"))
import render_side_by_side as RS  # noqa: E402  (mesh(), SoftRenderer)
sys.path.insert(0, str(D.ROOT / "ver_03_madis"))
import madis_common as mc  # noqa: E402


def nets_and_classes():
    ck = torch.load(D.CACHE / "recognizer.pt", weights_only=False)
    nets = []
    for sd in ck["state"]:
        n = R.Net(len(ck["classes"]))
        n.load_state_dict(sd)
        nets.append(n.eval())
    return nets, ck["classes"]


@torch.no_grad()
def best_window(nets, f, ci):
    best = (-1e9, 0, len(f))
    for L in SP.LENS:
        for s in range(0, max(1, len(f) - L + 1), SP.STRIDE):
            w = f[s:s + L]
            x = torch.tensor(R.resample(w))[None]
            xm = torch.tensor(R.resample(D.mirror(w)))[None]
            lp = np.mean([max(F.log_softmax(n(x), -1)[0, ci].item(), F.log_softmax(n(xm), -1)[0, ci].item())
                          for n in nets])
            if lp > best[0]:
                best = (lp, s, min(len(f), s + L))
    return best


# ------------------------------------------------ streams (body60, rh90, lh90 TRUE left) per source
def wlasl_stream(vid):
    d = D.WLASL_DIR / vid
    return (np.load(d / "body_rot6d.npy"), np.load(d / "hand_rot6d.npy"),
            D.left_true_from_model(np.load(d / "hand_rot6d_left_modelframe.npy")))


def h2s_stream(cid):
    env = lmdb.open(str(D.H2S_LMDB), readonly=True, lock=False)
    with env.begin() as t:
        z = np.load(io.BytesIO(t.get(cid.encode())), allow_pickle=True)
    L = min(len(z["segment"]), 304)
    return z["body_features"][:L, 66:126], z["right_features"][:L, :90], z["left_features"][:L, :90]


_SS = None


def signspark_stream(cid):
    global _SS
    if _SS is None:
        h = np.load(D.SS_T2P["hand"], allow_pickle=True).item()
        b = np.load(D.SS_T2P["body"], allow_pickle=True).item()
        _SS = (h["pred_poses"][0], b["pred_poses"][0], b["lengths"][0], list(b["video_names"][0]))
    hp, bp, L, names = _SS
    k = names.index(cid)
    n = int(L[k])
    return bp[k, :, :n].T, hp[2 * k + 1, :, :n].T, D.left_true_from_model(hp[2 * k, :, :n].T)


def madis_stream(cid):
    d = pickle.load(open(D.MADIS_PRED / f"{cid}.pkl", "rb"))
    body, rh, lh_r = mc.convert(np.asarray(d["feats_rst"], np.float64) * mc.STD + mc.MEAN)
    return body, rh, D.left_true_from_model(lh_r)


def row(stream, s, e, label, soft):
    body, rh, lh = (np.asarray(x)[s:e] for x in stream)
    V, J, faces = RS.mesh(body, lh, rh, flip_left=False)
    if soft[0] is None:
        soft[0] = RS.SoftRenderer(faces)
    tiles = []
    for t in np.linspace(0, len(V) - 1, 5).astype(int):
        img = soft[0].view(V[t], np.array([0.0, 0.25, 0.0]), 0.75, 300)
        tiles.append(img)
    img = np.concatenate(tiles, 1)
    bar = np.full((30, img.shape[1], 3), 45, np.uint8)
    import cv2
    cv2.putText(bar, label[:110], (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], 0)


def main():
    nets, classes = nets_and_classes()
    wl = D.load_wlasl()
    h2s = {c["id"]: c for c in D.load_h2s_gt()}
    ss = {c["id"]: c for c in D.load_signspark_t2p()}
    _, md = D.load_madis()
    md = {c["id"]: c for c in md}
    S_gt = np.load(D.CACHE / "spot_h2s_gt.npy")
    ids = list(h2s)
    out = D.CACHE / "words"
    out.mkdir(parents=True, exist_ok=True)
    for w in sys.argv[1:]:
        ci = classes.index(w)
        ref = max((c for c in wl if c["gloss"] == w), key=lambda c: best_window(nets, c["f"], ci)[0])
        cand = [k for k, i in enumerate(ids) if w in D.words(h2s[i]["text"]) and i in md]
        k = max(cand, key=lambda k: S_gt[k, ci])
        cid = ids[k]
        soft = [None]
        rows = [row(wlasl_stream(ref["id"]), 0, len(ref["f"]), f"1 real WLASL sign '{w.upper()}' (signer {ref['signer']})", soft)]
        for name, src, stream in (("2 REAL How2Sign signer", h2s, h2s_stream), ("3 SignSparK text-only", ss, signspark_stream),
                                  ("4 MaDiS", md, madis_stream)):
            lp, s, e = best_window(nets, src[cid]["f"], ci)
            rows.append(row(stream(cid), s, e, f"{name}: frames {s}-{e}, p('{w}') = {np.exp(lp):.2f}", soft))
        sheet = np.concatenate(rows, 0)
        title = np.full((34, sheet.shape[1], 3), 25, np.uint8)
        import cv2
        cv2.putText(title, f"'{w}' in: \"{h2s[cid]['text'][:95]}\"", (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 200), 1, cv2.LINE_AA)
        imageio.imwrite(out / f"{w}.png", np.concatenate([title, sheet], 0))
        print(f"{w}: {cid} -> {out / (w + '.png')}", flush=True)


if __name__ == "__main__":
    main()
