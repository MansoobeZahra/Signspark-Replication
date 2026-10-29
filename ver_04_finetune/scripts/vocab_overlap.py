"""Vocabulary overlap: 3D-LEX animated glosses vs SignAvatars-WLASL subset glosses vs How2Sign text.
Read-only on other folders; writes ver_04_finetune/outputs/vocab_overlap.json."""
import csv, json, re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SR = ROOT / "ver_03_sr_t2p" / "data"
audit = json.loads((SR / "3dlex" / "fbx_audit.json").read_text())
lex_all = sorted(audit)
lex_anim = sorted(g for g, a in audit.items() if a.get("varying_curves", 0) > 0)
g2i = json.loads((SR / "meta" / "gloss_to_ids.json").read_text())
conv = {p.name for p in (SR / "converted").iterdir()}
wl = {g: [i for i in ids if i in conv] for g, ids in g2i.items()}
wl = {g: v for g, v in wl.items() if v}

def words(split):
    c = Counter()
    with open(ROOT / "model_tests/data/soke_splits" / f"how2sign_realigned_{split}_preprocessed_fps.csv") as f:
        for r in csv.DictReader(f):
            c.update(re.findall(r"[a-z]+(?:'[a-z]+)?", r["SENTENCE"].lower()))
    return c

tr = words("train")
norm = lambda g: g.lower().replace("_", " ").replace("-", " ")
def in_h2s(g):
    w = norm(g).split()
    return all(tr[x] > 0 for x in w), min(tr[x] for x in w) if w else 0

both = sorted(set(lex_anim) & set(wl))
res = {
    "lex_total": len(lex_all), "lex_animated": len(lex_anim),
    "wlasl_glosses_converted": len(wl), "wlasl_clips_converted": sum(map(len, wl.values())),
    "wlasl_clips_per_gloss_median": sorted(map(len, wl.values()))[len(wl) // 2],
    "lex_anim_and_wlasl": both,
    "lex_anim_in_h2s_train": sum(in_h2s(g)[0] for g in lex_anim),
    "lex_anim_h2s_count_ge10": sum(in_h2s(g)[1] >= 10 for g in lex_anim),
    "lex_anim_h2s_count_ge100": sum(in_h2s(g)[1] >= 100 for g in lex_anim),
    "wlasl_in_h2s_train": sum(in_h2s(g)[0] for g in wl),
    "h2s_train_sentences_vocab": len(tr),
    "per_gloss": {g: {"wlasl_clips": len(wl.get(g, [])), "h2s_train_count": in_h2s(g)[1]} for g in lex_anim},
}
(ROOT / "ver_04_finetune/outputs/vocab_overlap.json").write_text(json.dumps(res, indent=1))
print({k: v for k, v in res.items() if k != "per_gloss"})
