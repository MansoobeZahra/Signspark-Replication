"""Step 2: seed / noise. Best-of-N by back-translation for text-only SignSparK.

Sentences: 50 How2Sign test clips = the 200-clip subset (eval_t2p.load_subset(200)) indices 0..48 plus 72
(G3FhmHz_7hs_6-10 "Now it's your responsibility to provide the healthy food.", a model_tests render clip),
+ 6 demo sentences.
Per sentence 32 candidates (seed 1000*i + j; hand and body share the seed), CFG 2.5, 10 steps, real length
(demo: words x median frames-per-word of the subset). Selection score: sentence chrF of the BT output against the
sentence. Best-of-N for N = 1, 8, 32 uses candidates 0..N-1.

Controls (no extra generation):
  chance  - the SAME selector (chrF vs sentence i) applied to the 32 candidates of a DIFFERENT sentence k = i+s
            (s = 1..49, averaged). Those candidates carry no information about sentence i, so whatever best-of-N
            gains there is selection on noise, not signing.
  pose    - 200-way retrieval (bt_common.retrieval) of the selected candidate among the subset's 200 real clips,
            with the same chance control; pose-oracle = the candidate closest to the real clip (upper bound of
            what seed choice alone can give).

    .venv-signspark/bin/python bt_bestofn.py gen      # ~4 min, outputs/bestofn/cands.npz
    .venv-signspark/bin/python bt_bestofn.py gen_ctrl # ~3 min, outputs/bestofn/ctrl_wrongtext.npz
    .venv-signspark/bin/python bt_bestofn.py score    # outputs/bestofn/results.json
"""

import json
import sys
import time

import numpy as np

import bt_common as bc

OUTD = bc.ROOT / "outputs" / "bestofn"
NC = 32
NS = (1, 8, 32)
TEST_IDX = list(range(49)) + [72]
DEMO = ["Hello, how are you?", "Thank you very much.", "What is your name?", "I am learning sign language.",
        "Nice to meet you.", "Where is the bathroom?"]
TAG = "<American> "
CTRL_SHIFT = 25


def sentences():
    from eval_t2p import load_subset
    sub = load_subset(200)
    fpw = float(np.median([c["L"] / len(c["text"].split()) for c in sub]))
    test = [sub[i] for i in TEST_IDX]
    demo = [{"id": f"demo{k}", "text": t, "L": int(np.clip(round(len(t.split()) * fpw), 40, 304))}
            for k, t in enumerate(DEMO)]
    return sub, test, demo, fpw


def gen(ctrl=False):
    """ctrl: the length- and noise-matched control. Test sentence i gets the TEXT of test sentence i+25 but its own
    length and its own seeds, so the only difference from the real candidates is the text. Shift 25, not 1: in the
    sorted subset, 18/50 neighbours (i, i+1) come from the same video (same topic and signer); at 25 none do."""
    import gen_common as g
    sub, test, demo, fpw = sentences()
    S = test if ctrl else test + demo
    src = [S[(i + CTRL_SHIFT) % len(S)] if ctrl else S[i] for i in range(len(S))]
    texts = [TAG + src[i]["text"] for i in range(len(S)) for _ in range(NC)]
    lens = [c["L"] for c in S for _ in range(NC)]
    seeds = [1000 * i + j for i in range(len(S)) for j in range(NC)]
    out = {}
    t0 = time.perf_counter()
    for stream, n_items in (("hand", 2), ("body", 1)):
        m, f = g.load(stream)
        g.TextBank(m).install()
        out[stream] = g.generate(m, f, texts, lens, seeds, n_items, scale=2.5, steps=10)
        del m, f
        print(f"{stream}: {time.perf_counter() - t0:.0f}s", flush=True)
    feats = np.empty(len(texts), dtype=object)
    for e in range(len(texts)):
        L = lens[e]
        feats[e] = bc.pred_feats(out["body"][e, 0, :L], out["hand"][e, 1, :L], out["hand"][e, 0, :L]).astype(np.float16)
    OUTD.mkdir(parents=True, exist_ok=True)
    np.savez(OUTD / ("ctrl_wrongtext.npz" if ctrl else "cands.npz"), feats=feats, texts=np.array([c["text"] for c in S]), ids=np.array([c["id"] for c in S]),
             lengths=np.array([c["L"] for c in S]), seeds=np.array(seeds).reshape(len(S), NC), fpw=fpw,
             gen_seconds=time.perf_counter() - t0)
    print(f"saved {len(texts)} candidates, {time.perf_counter() - t0:.0f}s")


def joints(fk, f240, left_true=True):
    """240-d features (left true SMPL-X-left) -> SoKeJoints dict."""
    from eval_t2p import smplx_in
    return fk(*smplx_in(f240[:, :60], f240[:, 150:], f240[:, 60:150], False))


def score():
    import soke_metric as sm
    from bt_validate import ortho
    t0 = time.perf_counter()
    sub, test, demo, _ = sentences()
    Z = np.load(OUTD / "cands.npz", allow_pickle=True)
    S, nt = len(Z["texts"]), len(test)
    feats = [f.astype(np.float32) for f in Z["feats"]]
    bt = bc.BT()
    refs = [bt.ref(t) for t in Z["texts"]]
    hyps = np.array(bt.translate(feats)).reshape(S, NC)
    print(f"BT {time.perf_counter() - t0:.0f}s", flush=True)

    # CH[k, j, i]: chrF of candidate j of sentence k against test sentence i
    CH = np.array([[[bc.sent_chrf(refs[i], hyps[k, j]) for i in range(nt)] for j in range(NC)] for k in range(nt)])
    print(f"chrF matrix {time.perf_counter() - t0:.0f}s", flush=True)

    # pose side: candidates of the 50 test sentences vs all 200 real clips of the subset
    fk = sm.SoKeJoints()
    gtR = [bc.retrieval_feats(joints(fk, bc.gt_feats(c))) for c in sub]
    candR = [bc.retrieval_feats(joints(fk, feats[k * NC + j])) for k in range(nt) for j in range(NC)]
    D = bc.dist_matrix(candR, gtR).reshape(nt, NC, len(sub))          # D[k, j, g]
    own = np.array(TEST_IDX)                                          # column of test sentence i's real clip
    C = np.load(OUTD / "ctrl_wrongtext.npz", allow_pickle=True)       # same length + seeds, text of sentence i+25
    cfeats = [f.astype(np.float32) for f in C["feats"]]
    hc = np.array(bt.translate(cfeats)).reshape(nt, NC)
    CC = np.array([[bc.sent_chrf(refs[i], hc[i, j]) for j in range(NC)] for i in range(nt)])
    Dc = bc.dist_matrix([bc.retrieval_feats(joints(fk, f)) for f in cfeats], gtR).reshape(nt, NC, len(sub))
    print(f"retrieval distances {time.perf_counter() - t0:.0f}s", flush=True)

    def rank(k, j, i, D=D):   # rank of real clip own[i] for candidate (k, j)
        return int((D[k, j] < D[k, j, own[i]]).sum() + 1)

    def rstats(r):
        r = np.asarray(r)
        return {"top1": float((r == 1).mean()), "top5": float((r <= 5).mean()), "median_rank": float(np.median(r)),
                "mean_rank": float(r.mean())}

    res = {"n_test": nt, "n_cands": NC, "chance": {"top1": 1 / 200, "top5": 5 / 200, "median_rank": 100.5},
           "best_of_n": {}}
    tref = refs[:nt]
    shifts = range(1, nt)
    for N in NS:
        sel = CH[np.arange(nt), :N, np.arange(nt)].argmax(1)                       # own pool, BT selector
        ps = [D[i, :N, own[i]].argmin() for i in range(nt)]                        # own pool, pose oracle
        row = {"bt_selected": bt.corpus(tref, [hyps[i, sel[i]] for i in range(nt)]),
               "mean_sent_chrf_selected": float(CH[np.arange(nt), sel, np.arange(nt)].mean()),
               "retrieval_bt_selected": rstats([rank(i, sel[i], i) for i in range(nt)]),
               "retrieval_pose_oracle": rstats([rank(i, ps[i], i) for i in range(nt)]),
               "bt_of_pose_oracle": bt.corpus(tref, [hyps[i, ps[i]] for i in range(nt)]),
               "motion_energy_selected": float(np.mean([bc.motion_energy(feats[i * NC + sel[i]]) for i in range(nt)]))}
        ch_bt, ch_ret, ch_po, ch_chrf = [], [], [], []
        for s in shifts:
            k = (np.arange(nt) + s) % nt
            sw = np.array([CH[k[i], :N, i].argmax() for i in range(nt)])
            ch_bt.append(bt.corpus(tref, [hyps[k[i], sw[i]] for i in range(nt)]))
            ch_chrf.append(np.mean([CH[k[i], sw[i], i] for i in range(nt)]))
            ch_ret.append([rank(k[i], sw[i], i) for i in range(nt)])
            pw = [D[k[i], :N, own[i]].argmin() for i in range(nt)]
            ch_po.append([rank(k[i], pw[i], i) for i in range(nt)])
        from scipy.stats import wilcoxon
        cs = CC[:, :N].argmax(1)                                                   # matched control, BT selector
        cp = [Dc[i, :N, own[i]].argmin() for i in range(nt)]
        row["ctrl_matched"] = {
            "bt_selected": bt.corpus(tref, [hc[i, cs[i]] for i in range(nt)]),
            "mean_sent_chrf_selected": float(CC[np.arange(nt), cs].mean()),
            "retrieval_bt_selected": rstats([rank(i, cs[i], i, Dc) for i in range(nt)]),
            "retrieval_pose_oracle": rstats([rank(i, cp[i], i, Dc) for i in range(nt)]),
            "wilcoxon_p_sent_chrf_own_gt_ctrl": float(wilcoxon(CH[np.arange(nt), sel, np.arange(nt)],
                                                               CC[np.arange(nt), cs], alternative="greater").pvalue)}
        row["bt_chance_wrong_pool"] = {m: float(np.mean([c[m] for c in ch_bt])) for m in ch_bt[0]}
        row["bt_chance_wrong_pool_p95"] = {m: float(np.percentile([c[m] for c in ch_bt], 95)) for m in ch_bt[0]}
        row["mean_sent_chrf_chance"] = float(np.mean(ch_chrf))
        row["retrieval_bt_selected_chance"] = rstats(np.concatenate(ch_ret))
        row["retrieval_pose_oracle_chance"] = rstats(np.concatenate(ch_po))
        res["best_of_n"][N] = row
        cm = row["ctrl_matched"]
        print(f"N={N:2} matched ctrl: B1 {cm['bt_selected']['bleu1']:.2f} chrF {cm['bt_selected']['chrf']:.2f} "
              f"sent-chrF own {row['mean_sent_chrf_selected']:.2f} vs {cm['mean_sent_chrf_selected']:.2f} "
              f"(p {cm['wilcoxon_p_sent_chrf_own_gt_ctrl']:.3f}) | BT-sel median rank {cm['retrieval_bt_selected']['median_rank']:.0f} "
              f"| pose-oracle top5 {cm['retrieval_pose_oracle']['top5']:.2f}", flush=True)
        print(f"N={N:2}: BT B1 {row['bt_selected']['bleu1']:.2f} (chance {row['bt_chance_wrong_pool']['bleu1']:.2f}) "
              f"chrF {row['bt_selected']['chrf']:.2f} (chance {row['bt_chance_wrong_pool']['chrf']:.2f}) | "
              f"retr. BT-sel median {row['retrieval_bt_selected']['median_rank']:.0f} "
              f"(chance {row['retrieval_bt_selected_chance']['median_rank']:.0f}) | pose-oracle top5 "
              f"{row['retrieval_pose_oracle']['top5']:.2f} (chance {row['retrieval_pose_oracle_chance']['top5']:.2f})",
              flush=True)

    # does the BT score track closeness to the real signing? within-sentence Spearman(chrF, -distance)
    from scipy.stats import spearmanr, wilcoxon
    rho = [spearmanr(CH[i, :, i], -D[i, :, own[i]])[0] for i in range(nt)]
    rho = [r for r in rho if np.isfinite(r)]
    res["spearman_chrf_vs_pose_closeness"] = {"mean": float(np.mean(rho)), "se": float(np.std(rho) / np.sqrt(len(rho))),
                                              "n": len(rho)}
    # all 32 candidates: own-sentence chrF vs chrF of the same hyps against other sentences
    own_ch = CH[np.arange(nt), :, np.arange(nt)].mean(1)
    oth_ch = np.array([np.delete(CH[i, :, :], i, axis=1).mean() for i in range(nt)])
    ctrl_ch = CC.mean(1)
    rk_own = np.array([[rank(i, j, i) for j in range(NC)] for i in range(nt)])
    rk_ctl = np.array([[rank(i, j, i, Dc) for j in range(NC)] for i in range(nt)])
    res["all_cands_vs_matched_ctrl"] = {
        "mean_sent_chrf_own": float(own_ch.mean()), "mean_sent_chrf_ctrl": float(ctrl_ch.mean()),
        "wilcoxon_p_chrf": float(wilcoxon(own_ch, ctrl_ch, alternative="greater").pvalue),
        "mean_rank_own": float(rk_own.mean()), "mean_rank_ctrl": float(rk_ctl.mean()),
        "wilcoxon_p_rank": float(wilcoxon(rk_ctl.mean(1), rk_own.mean(1), alternative="greater").pvalue),
        "retrieval_own": rstats(rk_own.ravel()), "retrieval_ctrl": rstats(rk_ctl.ravel()),
        "corpus_bt_own_cand0": bt.corpus(tref, list(hyps[:nt, 0])), "corpus_bt_ctrl_cand0": bt.corpus(tref, list(hc[:, 0]))}
    res["all_cands"] = {"mean_sent_chrf_own": float(own_ch.mean()), "mean_sent_chrf_other_sentences": float(oth_ch.mean()),
                        "wilcoxon_p_own_gt_other": float(wilcoxon(own_ch, oth_ch, alternative="greater").pvalue),
                        "retrieval": rstats([rank(i, j, i) for i in range(nt) for j in range(NC)]),
                        "distinct_hyps_per_sentence": float(np.mean([len(set(h)) for h in hyps[:nt]])),
                        "motion_energy": float(np.mean([bc.motion_energy(f) for f in feats[:nt * NC]]))}

    # baselines on the same 50 sentences
    gtf = [bc.gt_feats(c) for c in test]
    mean = ortho(np.concatenate([bc.gt_feats(c) for c in sub]).mean(0))
    stat = [np.repeat(mean[None], len(f), 0).astype(np.float32) for f in gtf]
    other = [gtf[(i + 1) % nt] for i in range(nt)]
    hs = bt.translate(stat)
    sc_own = np.array([bc.sent_chrf(tref[i], hs[i]) for i in range(nt)])
    sc_oth = np.array([np.mean([bc.sent_chrf(tref[k], hs[i]) for k in range(nt) if k != i]) for i in range(nt)])
    res["length_only_check"] = {"standing_still_sent_chrf_own_length_ref": float(sc_own.mean()),
                                "standing_still_sent_chrf_other_refs": float(sc_oth.mean())}
    statR = bc.retrieval_feats(joints(fk, stat[0][:64]))
    Ds = bc.dist_matrix([statR], gtR)[0]
    res["baselines"] = {
        "gt": {"bt": bt.corpus(tref, bt.translate(gtf)), "motion_energy": float(np.mean([bc.motion_energy(f) for f in gtf]))},
        "standing_still": {"bt": bt.corpus(tref, bt.translate(stat)),
                           "retrieval": rstats([(Ds < Ds[own[i]]).sum() + 1 for i in range(nt)])},
        "other_clip_gt": {"bt": bt.corpus(tref, bt.translate(other)),
                          "retrieval": rstats([(bc.dist_matrix([gtR[own[(i + 1) % nt]]], gtR)[0] <
                                                bc.dist_matrix([gtR[own[(i + 1) % nt]]], gtR)[0][own[i]]).sum() + 1
                                               for i in range(nt)])}}
    res["examples"] = [{"id": str(Z["ids"][i]), "ref": refs[i], "hyp_n1": hyps[i, 0],
                        "hyp_best32": hyps[i, CH[i, :, i].argmax()], "chrf_n1": float(CH[i, 0, i]),
                        "chrf_best32": float(CH[i, :, i].max()), "best32_index": int(CH[i, :, i].argmax()),
                        "pose_oracle_index": int(D[i, :, own[i]].argmin()),
                        "rank_n1": rank(i, 0, i), "rank_best32": rank(i, CH[i, :, i].argmax(), i)}
                       for i in range(nt)]
    res["demo"] = []
    for d in range(nt, S):
        sc = [bc.sent_chrf(refs[d], h) for h in hyps[d]]
        res["demo"].append({"text": str(Z["texts"][d]), "frames": int(Z["lengths"][d]), "hyp_n1": hyps[d, 0],
                            "chrf_n1": sc[0], "hyp_best32": hyps[d, int(np.argmax(sc))], "chrf_best32": max(sc),
                            "best32_index": int(np.argmax(sc)), "distinct_hyps": len(set(hyps[d]))})
    res["score_seconds"] = time.perf_counter() - t0
    res["gen_seconds"] = float(Z["gen_seconds"])
    np.save(OUTD / "hyps.npy", hyps)
    np.save(OUTD / "chrf_matrix.npy", CH)
    np.save(OUTD / "retrieval_dist.npy", D)
    np.save(OUTD / "ctrl_hyps.npy", hc)
    (OUTD / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("spearman_chrf_vs_pose_closeness", "all_cands", "all_cands_vs_matched_ctrl",
                                          "length_only_check", "baselines")}, indent=1,
                     default=float))
    for d in res["demo"]:
        print(f"DEMO {d['text']!r}: n1 {d['hyp_n1']!r} ({d['chrf_n1']:.1f}) | best32 {d['hyp_best32']!r} "
              f"({d['chrf_best32']:.1f})")
    print(f"score {res['score_seconds']:.0f}s")


if __name__ == "__main__":
    {"gen": gen, "gen_ctrl": lambda: gen(ctrl=True), "score": score}[sys.argv[1]]()
