"""Per-sign consensus + three-source cross-check (ASL Citizen x 3D-LEX x WLASL) -> the final dictionary sign.

1. Consensus (noise reduction across signers): the medoid's nearest ASL Citizen clips (well-tracked dominant hand,
   within the gloss's own closest quartile of distances, 3-8 of them; the number differs per sign) are DTW-aligned
   to the medoid on joint positions and their rotations averaged per frame and joint (weighted chordal mean, SVD
   projection). Independent tracking noise averages out; the shared sign stays. A sign whose clips disagree
   (variants, wide spread) gets few neighbours, so it is not blurred into a mix of variants.
2. Cross-check, word metric of ver_04 (DTW, trim 0.35, handedness-invariant), closed set = the dictionary glosses
   (variant pairs HOW1/HOW2, WHAT1/WHAT2 count as one word):
   * 3D-LEX (mocap, one signer) -> rank of its own gloss among the ASL Citizen signs;
   * each WLASL clip (same WiLoR+NLF pipeline as ASL Citizen) -> rank of its own gloss.
   Each reference votes (3D-LEX: own gloss in the top 3; WLASL: a majority of its clips do). A gloss is FLAGGED
   when every available reference disagrees, i.e. ASL Citizen stands alone.
3. Final sign: flagged and in 3D-LEX -> the 3D-LEX sign (rest frames trimmed, trim_rest); flagged, not in 3D-LEX -> the ASL Citizen clip closest
   to the WLASL clips; otherwise the consensus. (PAPER_AUDIT §6: the word metric is weak across sources, top-5
   ~0.55 at 735-way on real data, so a flag means "look at this one", not proof of a wrong sign.)

    CUDA_VISIBLE_DEVICES= ../ver-01/.venv-signspark/bin/python dictionary/consensus.py
Output: data/asl_citizen/dict/{consensus,final}/<GLOSS>/ (ver_02 streams + meta.json), crosscheck.json
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "dictionary"))
import aslc_build as ab  # noqa: E402
from compare_proto import dist_matrix, wm, wv  # noqa: E402  (CPU word metric, joints on CPU)

DICT = ab.OUT
WL = REPO / "data" / "wlasl"
STREAMS = ("body_rot6d.npy", "hand_rot6d.npy", "hand_rot6d_left_modelframe.npy")
LEX_NAME = {g: g.lower() for g in "BATHROOM HELP MEET PLEASE SCHOOL SEE SLOW SORRY TOMORROW WHERE WHO YOU YOUR".split()}
LEX_NAME.update(HOW1="how", HOW2="how")
WORD = lambda g: g.rstrip("12")  # noqa: E731  variant pairs are one word


def load(d):
    return [np.load(d / k) for k in STREAMS]


def feat_pair(streams):
    J = wm.stream_joints(*streams)
    return (wv.feats(wv.trim(J, 0.35), "full").astype(np.float32),
            wv.feats(wv.trim(wm.mirror_joints(J), 0.35), "full").astype(np.float32))


def trim_rest(streams, frac=0.2, pad=4):
    """3D-LEX recordings start and end in a long rest pose (SEE: 141 frames, mostly rest). Keep the span where
    either wrist is above frac of its rise over the clip (relative to the pelvis), +-pad frames, so a stitched
    sentence does not sit in rest between signs."""
    J = wm.stream_joints(*streams)
    h = (J["body"][:, [20, 21], 1] - J["body"][:, :1, 1]).max(1)
    idx = np.flatnonzero(h >= h.min() + frac * (h.max() - h.min()))
    a, b = max(0, idx[0] - pad), min(len(h), idx[-1] + 1 + pad)
    return [x[a:b] for x in streams], (int(a), int(b), len(h))


def frame_feats(streams):
    """Per-frame joint positions (pelvis-relative body + wrist-relative hands), for DTW alignment."""
    J = wm.stream_joints(*streams)
    b = J["body"][:, wm.sm.BODY_IDX] - J["body"][:, :1]
    x = np.concatenate([b, J["lhand"] - J["lhand"][:, :1], J["rhand"] - J["rhand"][:, :1]], 1)
    return x.reshape(len(x), -1)


def dtw_path(A, B):
    C = np.linalg.norm(A[:, None] - B[None], axis=-1)
    n, m = C.shape
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        k = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
        i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
    return path[::-1]


def chordal_mean(Rs, w):
    """Weighted chordal L2 mean of rotations Rs (N, J, 3, 3) -> (J, 3, 3)."""
    M = np.einsum("n,njab->jab", w / w.sum(), Rs)
    U, _, Vt = np.linalg.svd(M)
    S = np.ones((len(M), 3))
    S[:, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ (S[:, :, None] * Vt)


def consensus(med, nbrs, wts):
    """Average neighbour clips onto the medoid's timeline. Mirrored-better neighbours were mirrored at build time
    already (dominance), so plain alignment is used here."""
    fm = frame_feats(med)
    Rm = [ab.from6(s) for s in med]
    acc = [[[r[t]] for t in range(len(r))] for r in Rm]                 # per stream, per medoid frame
    wacc = [[[1.0] for _ in range(len(r))] for r in Rm]
    for s, w in zip(nbrs, wts):
        Rn = [ab.from6(x) for x in s]
        for i, j in dtw_path(fm, frame_feats(s)):
            for k in range(3):
                acc[k][i].append(Rn[k][j])
                wacc[k][i].append(w)
    out = []
    for k in range(3):
        out.append(np.stack([chordal_mean(np.stack(a), np.array(w)) for a, w in zip(acc[k], wacc[k])]))
    return [ab.r6(R) for R in out]


def closed_rank(Fq, Fq_m, own, bank_glosses, bank_F):
    d = np.minimum(dist_matrix([Fq], bank_F), dist_matrix([Fq_m], bank_F))[0]
    by_word = {}
    for g, x in zip(bank_glosses, d):
        by_word[WORD(g)] = min(by_word.get(WORD(g), np.inf), x)
    words = sorted(by_word, key=by_word.get)
    return words.index(WORD(own)) + 1, words[:3], by_word


def index_glosses():
    return json.loads((DICT / "index.json").read_text())["glosses"]


def wlasl_clips(st):
    """Build the WLASL clips through exactly the ASL Citizen pipeline (aslc_build.load_clip)."""
    fast = torch.load(WL / "sltk" / "fast.pt", weights_only=False) if (WL / "sltk" / "fast.pt").exists() else {}
    out = {}
    for v in sorted((WL / "videos").rglob("*.mp4")):
        if not (WL / "sltk" / "nlf" / f"{v.stem}.h5").exists() or not (WL / "sltk" / "wilor" / f"{v.stem}.h5").exists():
            continue
        g = v.parent.name.upper().replace("_", "").replace(" ", "")
        seg = fast.get(v.stem)
        s, meta = ab.load_clip(v.stem, seg.numpy() if seg is not None else None, st["rh"], st["lh_rframe"],
                               feats=WL / "sltk")
        if s is None:
            continue
        s = ab.apply_contact(s, meta, g if g in index_glosses() else g + "1")
        d = WL / DICT.name / "clips" / g / v.stem
        ab.save_streams(d, s, {**meta, "gloss": g, "source": "WLASL"})
        out.setdefault(g, []).append(d)
    return out


def one_consensus(g, e):
    """Consensus of gloss g (index entry e) -> streams, saved to consensus/<g>; returns (streams, n clips)."""
    dirs = {c: DICT / "clips" / g / c for c in e["clips"]}
    cov = {c: json.loads((d / "meta.json").read_text())["hand_frames"]["rh"] for c, d in dirs.items()}
    streams = {c: load(d) for c, d in dirs.items()}
    F = {c: feat_pair(s) for c, s in streams.items()}
    med = e["medoid"]
    others = [c for c in e["clips"] if c != med and cov[c] >= 0.6 and c not in e.get("short_clips", [])
              and c not in e.get("no_contact_clips", [])]
    out = DICT / "consensus" / g
    out.mkdir(parents=True, exist_ok=True)
    if not others:                                   # single usable clip: the consensus is the medoid itself
        for k, x in zip(STREAMS, streams[med]):
            np.save(out / k, x)
        (out / "meta.json").write_text(json.dumps({"gloss": g, "medoid": med, "neighbours": [], "fps": ab.FPS}))
        return streams[med], 1
    d = np.minimum(dist_matrix([F[med][0]], [F[c][0] for c in others]),
                   dist_matrix([F[med][1]], [F[c][0] for c in others]))[0]
    allrow = np.minimum(dist_matrix([F[med][0]], [F[c][0] for c in e["clips"] if c != med]),
                        dist_matrix([F[med][1]], [F[c][0] for c in e["clips"] if c != med]))[0]
    q = np.quantile(allrow, 0.25)
    order = np.argsort(d)
    pick = [i for i in order if d[i] <= q][:8]
    pick = pick if len(pick) >= 3 else list(order[:3])
    tau = np.median(allrow)
    wts = np.exp(-d[pick] / tau)
    cs = consensus(streams[med], [streams[others[i]] for i in pick], wts)
    for k, x in zip(STREAMS, cs):
        np.save(out / k, x)
    (out / "meta.json").write_text(json.dumps({"gloss": g, "medoid": med, "neighbours": [others[i] for i in pick],
                                               "weights": wts.round(3).tolist(), "fps": ab.FPS}))
    return cs, len(pick) + 1


def main():
    index = json.loads((DICT / "index.json").read_text())["glosses"]
    st = np.load(REPO / "outputs" / "static_How2Sign.npz")
    glosses = sorted(index)
    report = {}
    # 1. consensus per gloss
    cons_F = {}
    for g in glosses:
        cs, n = one_consensus(g, index[g])
        cons_F[g] = feat_pair(cs)
        report[g] = {"consensus_n": n, "medoid": index[g]["medoid"]}
        print(f"{g:13} consensus of {n} clips", flush=True)
    bank_g = glosses
    bank_F = [cons_F[g][0] for g in bank_g]
    # 2a. 3D-LEX
    for g, name in LEX_NAME.items():
        r, top, _ = closed_rank(*feat_pair(load(wm.LEX / name)), g, bank_g, bank_F)
        report[g]["lex_rank"], report[g]["lex_top3"] = r, top
    # 2b. WLASL
    wl = wlasl_clips(st)
    wl_F = {g: [feat_pair(load(d)) for d in ds] for g, ds in wl.items()}
    for g in glosses:
        w = WORD(g)
        if w not in wl_F:
            continue
        ranks = [closed_rank(f, fm, g, bank_g, bank_F)[0] for f, fm in wl_F[w]]
        report[g]["wlasl_ranks"] = ranks
    # 3. flag + final
    for g in glosses:
        r = report[g]
        # each available reference votes; ASL Citizen is kept when at least one independent source agrees with it
        # (3D-LEX alone disagreeing while WLASL agrees is usually 3D-LEX's own variant, e.g. HOW)
        votes = []
        if "lex_rank" in r:
            votes.append(r["lex_rank"] <= 3)
        if r.get("wlasl_ranks"):
            votes.append(np.mean(np.array(r["wlasl_ranks"]) <= 3) > 0.5)
        r["flagged"] = bool(votes) and not any(votes)
        dst = DICT / "final" / g
        if dst.exists():
            shutil.rmtree(dst)
        if r["flagged"] and g in LEX_NAME:
            src, r["final"] = wm.LEX / LEX_NAME[g], f"3D-LEX:{LEX_NAME[g]}"
        elif r["flagged"] and WORD(g) in wl_F:
            ref = [f for f, _ in wl_F[WORD(g)]]
            cands = [c for c in index[g]["clips"]
                     if c not in index[g].get("short_clips", []) + index[g].get("no_contact_clips", [])
                     and json.loads((DICT / "clips" / g / c / "meta.json").read_text())["hand_frames"]["rh"] >= 0.6]
            cf = [feat_pair(load(DICT / "clips" / g / c)) for c in cands]
            dd = np.minimum(dist_matrix([a for a, _ in cf], ref), dist_matrix([b for _, b in cf], ref)).mean(1)
            best = cands[int(np.argmin(dd))]
            src, r["final"] = DICT / "clips" / g / best, f"ASLCitizen-closest-to-WLASL:{best}"
        else:
            src, r["final"] = DICT / "consensus" / g, "consensus"
        dst.mkdir(parents=True)
        meta = {"gloss": g, "source": r["final"], "fps": 24}
        if src.parent == wm.LEX:
            x, meta["trim_rest"] = trim_rest(load(src))
            for k, v in zip(STREAMS, x):
                np.save(dst / k, v)
        else:
            for k in STREAMS:
                shutil.copy(src / k, dst / k)
        (dst / "meta.json").write_text(json.dumps(meta))
    (DICT / "crosscheck.json").write_text(json.dumps(report, indent=1))
    print(f"\n{'gloss':13} {'cons':>4} {'lex':>4}  {'wlasl ranks':16} flag  final")
    for g in glosses:
        r = report[g]
        print(f"{g:13} {r['consensus_n']:4} {str(r.get('lex_rank', '-')):>4}  {str(r.get('wlasl_ranks', '-')):16} "
              f"{'FLAG' if r['flagged'] else '    '}  {r['final']}")


if __name__ == "__main__":
    main()
