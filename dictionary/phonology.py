"""Linguistic check of the dictionary against ASL-LEX 2.0 phonology (handshape, location, one/two hands).

ASL Citizen's vocabulary is ASL-LEX (its CSV carries the ASL-LEX Code), so every gloss has a linguist's coding.
Three attributes are measured on the SMPL-X joints of a sign (dominant = signer's right, canonicalised at build):
  * handshape: which of index/middle/ring/pinky are extended (straightness tip-to-knuckle / chain length, median
    over the active frames), compared with the pattern of the ASL-LEX handshape. Only handshapes with an
    unambiguous extended/closed pattern are checked (curved and bent shapes such as c, o, baby_o are skipped);
  * major location: Head / Body / Hand / Neutral, from how close the dominant hand comes to the face, the torso
    front and the other hand;
  * sign type: one-handed vs two-handed, from how much the non-dominant hand rises and moves.
The decision thresholds are not hand-tuned: they are fitted on ASL Citizen's train-split clips (ASL-LEX labels)
and scored on its test-split clips, whose signers are unseen. The fitted checks are then applied to every final
dictionary sign; a disagreement is reported with the gloss's own clip agreement rate, so a check that also fails
on most real recordings of that gloss reads as "measurement unsure", not "wrong sign".

Arbitration (--apply): a final sign that replaced the consensus after the cross-check (3D-LEX, or the ASL Citizen
clip closest to WLASL) is kept only if it agrees with ASL-LEX at least as often as the consensus does; otherwise
the consensus goes back into final/ (ASL-LEX as the deciding vote between two disagreeing recordings).

    CUDA_VISIBLE_DEVICES= .venv-sltk/bin/python dictionary/phonology.py [--apply]
Output: outputs/dictionary/phonology_check.json
"""
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "dictionary"))
from compare_proto import wm  # noqa: E402  (CPU SMPL-X joints)

DICT = Path(os.environ.get("ASLC_DICT", REPO / "data" / "asl_citizen" / "dict"))   # ASLC_DICT: A/B builds
LEX_CSV = REPO / "data" / "asl_lex" / "signdata.csv"
SPLITS = REPO / "data" / "asl_citizen" / "ASL_Citizen" / "splits"
OUT = REPO / "outputs" / "dictionary" / f"phonology_check_{DICT.name}.json"
CACHE = DICT / "phonology_feats.npz"

FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]  # thumb..pinky
# extended index/middle/ring/pinky per ASL-LEX handshape (unambiguous shapes only)
EXT = {"1": "1000", "l": "1000", "g": "1000", "flat_1": "1000", "d": "1000",
       "open_b": "1111", "5": "1111", "closed_b": "1111", "4": "1111", "flatspread_5": "1111", "flat_b": "1111",
       "a": "0000", "s": "0000", "t": "0000",
       "h": "1100", "v": "1100", "open_h": "1100", "flat_h": "1100", "3": "1100", "r": "1100", "flat_v": "1100",
       "y": "0001", "i": "0001", "w": "1110", "f": "0111", "open_8": "1011", "8": "1011", "horns": "1001"}
LOCS = ("Head", "Body", "Hand", "Neutral")
FACE = [15, 22, *range(55, 60), *range(76, 144)]         # head, jaw, nose/eyes/ears, face landmarks
TORSO = [3, 6, 9, 12, 13, 14]                            # spine1-3, neck, collars (moved to the torso front)


def lex_table():
    lex = {x["Code"]: x for x in csv.DictReader(open(LEX_CSV, encoding="latin-1"))}
    out, split = {}, {}
    for s in ("train", "val", "test"):
        for r in csv.DictReader(open(SPLITS / f"{s}.csv")):
            if r["ASL-LEX Code"] in lex:
                out[r["Gloss"]] = lex[r["ASL-LEX Code"]]
            split[Path(r["Video file"]).stem] = s
    return out, split


def feats(d):
    """Joint features of one sign folder: 5 finger straightness, 5 location, 4 non-dominant hand activity."""
    J = wm.stream_joints(*wm.load_streams(d))
    B, R, L = J["body"], J["rhand"], J["lhand"]
    y = R[:, 0, 1]
    act = y >= np.quantile(y, 0.4)                       # active (raised) part of the sign
    st = []
    for ch in FINGERS:
        seg = np.linalg.norm(np.diff(R[:, ch], axis=1), axis=-1).sum(1)
        st.append(np.median((np.linalg.norm(R[:, ch[-1]] - R[:, ch[0]], axis=-1) / (seg + 1e-9))[act]))
    pts = R[:, [0, 4, 8, 12, 16, 20, 9]]                 # wrist, fingertips, palm centre
    dmin = lambda Q: np.linalg.norm(pts[:, :, None] - Q[:, None], axis=-1).min((1, 2))  # noqa: E731
    torso = B[:, TORSO] + np.array([0, 0, 0.09])
    d_face, d_torso, d_hand = dmin(B[:, FACE]), dmin(torso), dmin(L)
    h = (y - B[:, 12, 1]) / (B[:, 12, 1] - B[:, 0, 1])     # wrist height relative to the neck, torso units
    loc = [np.log(d_face[act].min() + 1e-3), np.log(d_torso[act].min() + 1e-3), np.log(d_hand[act].min() + 1e-3),
           np.median(h[act]), np.median((R[:, 0, 2] - B[:, 9, 2])[act])]
    ly = L[:, 0, 1] - B[:, 0, 1]
    lv = np.linalg.norm(np.diff(L[:, 0], axis=0), axis=-1)
    nd = [ly.max(), np.quantile(ly, 0.8) - np.quantile(ly, 0.2), lv.sum() / max(len(lv), 1), np.log(d_hand.min() + 1e-3)]
    return np.array(st + loc + nd, np.float32)


def collect():
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        return list(z["gloss"]), list(z["stem"]), z["x"]
    gl, st, xs = [], [], []
    for m in sorted(DICT.glob("clips/*/*/meta.json")):
        try:
            xs.append(feats(m.parent))
        except Exception as e:  # noqa: BLE001  a broken clip must not stop the check
            print("skip", m.parent, e)
            continue
        gl.append(m.parent.parent.name)
        st.append(m.parent.name)
    np.savez(CACHE, gloss=np.array(gl), stem=np.array(st), x=np.stack(xs))
    return gl, st, np.stack(xs)


def labels(row):
    hs = EXT.get(row["Handshape.2.0"])
    loc = row["MajorLocation.2.0"]
    return {"handshape": hs, "location": loc if loc in LOCS else None,
            "two_handed": row["SignType.2.0"] != "OneHanded"}


class Checks:
    """Fitted on train-split clips: 4 per-finger extended classifiers, location (4-way), two-handedness."""

    def __init__(self, X, Y):
        lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))  # noqa: E731
        self.fing = []
        hs = np.array([y["handshape"] is not None for y in Y])
        for f in range(4):
            yy = np.array([int(y["handshape"][f]) for y in np.array(Y)[hs]])
            self.fing.append(lr().fit(X[hs][:, 1:5], yy))  # all four finger straightnesses: neighbours co-vary
        lc = np.array([y["location"] is not None for y in Y])
        self.loc = lr().fit(X[lc][:, 5:10], [y["location"] for y in np.array(Y)[lc]])
        self.two = lr().fit(X[:, 10:14], [y["two_handed"] for y in Y])

    def predict(self, X):
        hs = ["".join(str(int(m.predict(X[i:i + 1, 1:5])[0])) for m in self.fing) for i in range(len(X))]
        return [{"handshape": h, "location": loc, "two_handed": bool(t)}
                for h, loc, t in zip(hs, self.loc.predict(X[:, 5:10]), self.two.predict(X[:, 10:14]))]


def agree(p, y):
    return {k: (None if y[k] is None else p[k] == y[k]) for k in y}


def rate(rows, k):
    v = [r[k] for r in rows if r[k] is not None]
    return (float(np.mean(v)), len(v)) if v else (None, 0)


def n_agree(a):
    return sum(v is True for v in a.values())


def main():
    apply = "--apply" in sys.argv
    lex, split = lex_table()
    gl, st, X = collect()
    keep = [i for i, g in enumerate(gl) if g in lex]
    gl, st, X = [gl[i] for i in keep], [st[i] for i in keep], X[keep]
    Y = [labels(lex[g]) for g in gl]
    sp = np.array([split.get(s, "train") for s in st])
    tr = np.flatnonzero(sp == "train")
    te = np.flatnonzero(sp == "test")
    ck = Checks(X[tr], [Y[i] for i in tr])
    P = ck.predict(X)
    A = [agree(p, y) for p, y in zip(P, Y)]
    res = {"n_glosses": len(set(gl)), "n_clips": len(gl),
           "test_clip_agreement_unseen_signers": {k: rate([A[i] for i in te], k)
                                                  for k in ("handshape", "location", "two_handed")},
           "majority_baseline_test": {}, "final": {}}
    for k in ("handshape", "location", "two_handed"):
        v = [str(Y[i][k]) for i in te if Y[i][k] is not None]
        res["majority_baseline_test"][k] = max(v.count(c) for c in set(v)) / len(v) if v else None
    by_gloss = {}
    for g, a in zip(gl, A):
        by_gloss.setdefault(g, []).append(a)
    for f in sorted((DICT / "final").glob("*/meta.json")):
        g = f.parent.name
        if g not in lex:
            continue
        p = ck.predict(feats(f.parent)[None])[0]
        y = labels(lex[g])
        a = agree(p, y)
        res["final"][g] = {"source": json.loads(f.read_text())["source"], "asl_lex": y, "measured": p,
                           "agree": a, "clip_agreement": {k: rate(by_gloss.get(g, []), k)[0] for k in a},
                           "flag": [k for k, v in a.items() if v is False]}
        src = json.loads(f.read_text())["source"]
        cons = DICT / "consensus" / g
        if src != "consensus" and cons.exists():
            ac = agree(ck.predict(feats(cons)[None])[0], y)
            res["final"][g]["consensus_agree"] = ac
            if n_agree(ac) > n_agree(a):
                res["final"][g]["arbitration"] = "consensus agrees better with ASL-LEX -> consensus"
                if apply:
                    for k in ("body_rot6d.npy", "hand_rot6d.npy", "hand_rot6d_left_modelframe.npy"):
                        shutil.copy(cons / k, f.parent / k)
                    f.write_text(json.dumps({"gloss": g, "source": "consensus", "fps": 24,
                                             "replaced": src, "reason": "ASL-LEX phonology arbitration"}))
            else:
                res["final"][g]["arbitration"] = f"keep {src}"
    OUT.write_text(json.dumps(res, indent=1, default=str))
    print(f"{res['n_glosses']} glosses, {res['n_clips']} clips")
    print("test-split clip agreement with ASL-LEX (unseen signers):", res["test_clip_agreement_unseen_signers"])
    print("majority-class baseline:", res["majority_baseline_test"])
    print(f"\n{'gloss':13} {'source':34} {'handshape lex/meas':20} {'location lex/meas':22} {'2-hand lex/meas':16} clips-agree h/l/t")
    for g, r in res["final"].items():
        y, p, c = r["asl_lex"], r["measured"], r["clip_agreement"]
        fmt = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
        print(f"{g:13} {r['source'][:34]:34} {str(y['handshape']):>5}/{p['handshape']:5}{'!' if r['agree']['handshape'] is False else ' ':9}"
              f"{str(y['location']):>8}/{p['location']:8}{'!' if r['agree']['location'] is False else ' ':5}"
              f"{str(y['two_handed'])[0]}/{str(p['two_handed'])[0]}{'!' if r['agree']['two_handed'] is False else ' ':12}"
              f"{fmt(c['handshape'])}/{fmt(c['location'])}/{fmt(c['two_handed'])}")
    for g, r in res["final"].items():
        if "arbitration" in r:
            print(f"arbitration {g:10} {r['arbitration']}{' (applied)' if apply and 'consensus' in r['arbitration'] else ''}")


if __name__ == "__main__":
    main()
