"""Full dictionary after the sharded build (aslc_build.py --shard k/n): medoid + consensus + contact snap for every
ASL Citizen gloss, with the DTW on the GPU (same length-normalised DTW as compare_proto.dist_matrix).

The 27 demo glosses keep their cross-checked finals (consensus.py: 3D-LEX / WLASL votes, phonology arbitration).
Every other gloss gets final/<GLOSS> = its consensus (source "consensus (not cross-checked)") + ASL-LEX contact
snap. Re-ranking the demo glosses against a 2.7k-sign bank would change their flags, so they are not touched.

    .venv-sltk/bin/python dictionary/full_merge.py --shard k/n     # medoids + consensus + final for glosses k::n
    .venv-sltk/bin/python dictionary/full_merge.py --gather        # -> index_full.json (+ failures from shards)
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aslc_build as ab  # noqa: E402
import compare_proto as cp  # noqa: E402
import consensus as co  # noqa: E402
import contact_snap as cs  # noqa: E402

DICT = ab.OUT
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DEMO = set(json.loads((DICT / "index.json").read_text())["glosses"])
MED = DICT / "medoids"


def dist_matrix_gpu(P, G):
    Gt = torch.tensor(np.stack(G), device=DEV)
    D = np.zeros((len(P), len(G)))
    for i, p in enumerate(P):
        C = (torch.tensor(p, device=DEV)[None, :, None] - Gt[:, None]).norm(dim=-1).mean(-1)
        D[i] = (cp.wm.bc.dtw_batch(C) / 128).cpu().numpy()
    return D


def run_shard(k, n):
    cp.dist_matrix = co.dist_matrix = dist_matrix_gpu        # ab.medoids imports it from compare_proto at call time
    codes = cs.lex_codes()
    # gloss = path between clips/ and the clip folder (a few ASL Citizen glosses contain "/", e.g. HURDLE/TRIP1)
    allg = {str(m.parent.parent.relative_to(DICT / "clips")) for m in (DICT / "clips").glob("**/meta.json")}
    glosses = sorted(g for g in allg if g not in DEMO)[k::n]
    MED.mkdir(exist_ok=True)
    for i, g in enumerate(glosses):
        mf = MED / f"{g.replace('/', '__')}.json"
        if mf.exists():
            continue
        try:
            e = {"clips": sorted(p.parent.name for p in (DICT / "clips" / g).glob("*/meta.json"))}
            ab.medoids({g: e})
            stm, nc = co.one_consensus(g, e)
            dst = DICT / "final" / g
            if dst.exists():
                shutil.rmtree(dst)
            dst.mkdir(parents=True)
            for f in co.STREAMS:
                shutil.copy(DICT / "consensus" / g / f, dst / f)
            meta = {"gloss": g, "source": "consensus (not cross-checked)", "consensus_n": nc, "fps": ab.FPS}
            r = codes.get(g)
            if cs.is_contact(r):
                out, info = cs.snap(cs.wm.load_streams(dst), r["MajorLocation.2.0"])
                info.update(major=r["MajorLocation.2.0"], minor=r["MinorLocation.2.0"])
                meta["contact_snap"] = info
                if out is not None:
                    np.save(dst / "body_rot6d.npy", out[0])
            (dst / "meta.json").write_text(json.dumps(meta))
            e["consensus_n"] = nc
        except Exception as ex:  # report, keep going
            e = {"error": f"{type(ex).__name__}: {ex}"}
            print(g, e["error"], flush=True)
        mf.write_text(json.dumps(e))
        if i % 50 == 0:
            print(f"shard {k}/{n}: {i}/{len(glosses)} {g}", flush=True)
    print(f"shard {k}/{n} done: {len(glosses)} glosses", flush=True)


def gather():
    index = json.loads((DICT / "index.json").read_text())["glosses"]
    errors = {}
    for f in sorted(MED.glob("*.json")):
        e = json.loads(f.read_text())
        (errors if "error" in e else index)[f.stem.replace("__", "/")] = e
    fails = {}
    for f in DICT.glob("shard_*of*.json"):
        fails.update(json.loads(f.read_text())["failed"])
    (DICT / "index_full.json").write_text(json.dumps({"glosses": index, "failed": fails, "gloss_errors": errors},
                                                     indent=1))
    print(f"{len(index)} glosses ({len(DEMO)} cross-checked demo), {sum(len(e['clips']) for e in index.values())} "
          f"clips, {len(fails)} failed clips, {len(errors)} gloss errors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default=None)
    ap.add_argument("--gather", action="store_true")
    a = ap.parse_args()
    if a.gather:
        gather()
    else:
        run_shard(*map(int, (a.shard or "0/1").split("/")))


if __name__ == "__main__":
    main()
