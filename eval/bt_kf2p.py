"""Back-translation (SignSparK-BT) of saved evaluate.py generations: GT vs SLERP vs SignSparK KF2P, same clips.

Paper Tab. 4 reports B-T BLEU-4 for KF2P and SLERP. BLEU-4 on 200 clips is mostly 0 (ver_03_gf_t2p/REPORT.md
§2.1), so chrF and BLEU-1 are the main numbers here, with a paired test on sentence-level chrF.
BT model + wrapper: ../ver_03_gf_t2p/bt_common.py (authors' SignSparK_BT, unmodified; How2Sign weights only).

    python eval/bt_kf2p.py --dataset How2Sign --runs slerp_How2Sign_kf2p released_How2Sign_kf2p_s10 ...
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[1]
GF = REPO.parent / "ver_03_gf_t2p"
sys.path.insert(0, str(GF))
sys.path.insert(1, str(REPO / "eval"))
import bt_common as bc  # noqa: E402
import evaluate as ev  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="How2Sign")
    ap.add_argument("--runs", nargs="+", required=True, help="outputs/eval/<run>_pred.npz names")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.dataset != "How2Sign":
        sys.exit("only the How2Sign BT model is downloaded")

    clips = ev.read_clips(ev.lmdb_path(REPO / "data" / "lmdb", "test", a.dataset), 200)
    raw = [c["text"].split("> ", 1)[1] for c in clips]  # drop the "<American> " language tag
    bt = bc.BT()
    refs = [bt.ref(t) for t in raw]
    feats = {"gt": [bc.gt_feats(c) for c in clips]}
    sb, srh, slh, _ = ev.static_pose(REPO / "data" / "lmdb", a.dataset)
    feats["static"] = [bc.pred_feats(np.repeat(sb[None], c["L"], 0), np.repeat(slh[None], c["L"], 0),
                                     np.repeat(srh[None], c["L"], 0)) for c in clips]
    for r in a.runs:
        z = np.load(REPO / "outputs" / "eval" / f"{r}_pred.npz")
        assert [str(i) for i in z["ids"]] == [c["id"] for c in clips], r
        feats[r] = [bc.pred_feats(z["body"][k, :c["L"]], z["hand"][k, 0, :c["L"]], z["hand"][k, 1, :c["L"]])
                    for k, c in enumerate(clips)]

    res, sent = {}, {}
    for name, f in feats.items():
        hyps = bt.translate(f)
        res[name] = bt.corpus(refs, hyps)
        sent[name] = np.array([bc.sent_chrf(r, h) for r, h in zip(refs, hyps)])
        res[name]["examples"] = hyps[:5]
        print(f"{name:32} " + " ".join(f"{k} {res[name][k]:.2f}" for k in ("bleu1", "bleu4", "chrf", "rouge")),
              flush=True)
    names = list(feats)
    res["paired_sentence_chrf"] = {f"{x}>{y}": {"frac": float(np.mean(sent[x] > sent[y])),
                                               "p": float(wilcoxon(sent[x], sent[y], alternative="greater").pvalue)}
                                   for x in names for y in names if x != y and y in ("slerp_" + a.dataset + "_kf2p",)}
    res["refs_examples"] = refs[:5]
    out = Path(a.out or REPO / "outputs" / "eval" / f"bt_kf2p_{a.dataset}.json")
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res["paired_sentence_chrf"], indent=1))


if __name__ == "__main__":
    main()
