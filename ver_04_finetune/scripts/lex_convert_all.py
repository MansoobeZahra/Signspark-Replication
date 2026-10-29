"""Convert all animated 3D-LEX signs with ver_03_sr_t2p's verified converter (imported, unmodified), into
ver_04_finetune/data/3dlex/<gloss>/ (same stream files as ver_03_sr_t2p/data/3dlex/converted).
Difference from lex3d_convert.convert_sign: output folder, and handedness is decided automatically
(mirror when the left wrist travels further than the right), recorded in meta.json.
    ../ver-01/.venv-signspark/bin/python scripts/lex_convert_all.py [--signs a b] [--limit N]"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ver_03_sr_t2p"))
import lex3d_convert as lc  # noqa: E402
from lex3d_io import load_fbx  # noqa: E402

OUT = ROOT / "ver_04_finetune" / "data" / "3dlex"


def wrist_path(la, side):
    p = la["P"][:, la["names"].index(f"{side}Hand")]
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def convert(g):
    sk = load_fbx(lc.fbx_bytes(g))
    if np.ptp(sk.q_local, axis=0).max() < 1e-4:
        return None
    la = lc.lex_arrays(sk)
    pl, pr = wrist_path(la, "Left"), wrist_path(la, "Right")
    mir = pl > 1.2 * pr                                       # clearly left-dominant -> mirror to right-dominant
    if mir:
        la = lc.mirror(la)
    L, A, _ = lc.convert_arrays(la)
    body, rh, lh = lc.to_streams(L)
    o = OUT / g
    o.mkdir(parents=True, exist_ok=True)
    np.save(o / "body_rot6d.npy", body); np.save(o / "hand_rot6d.npy", rh); np.save(o / "hand_rot6d_left_modelframe.npy", lh)
    meta = {"gloss": g, "fps": la["fps"], "frames": int(len(L)), "mirrored_auto": bool(mir),
            "wrist_path_m": {"left": pl / 100, "right": pr / 100}, "manual_mirror_in_sr_t2p": g in lc.MIRROR}
    (o / "meta.json").write_text(json.dumps(meta))
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--signs", nargs="*"); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    audit = json.loads((lc.LEX / "fbx_audit.json").read_text())
    gl = a.signs or sorted(g for g, v in audit.items() if v["varying_curves"] > 0)
    gl = gl[: a.limit] if a.limit else gl
    t0 = time.time(); n = 0
    for g in gl:
        try:
            m = convert(g)
        except Exception as e:  # noqa: BLE001
            print(f"{g}: FAILED {e}", flush=True); continue
        n += m is not None
    print(f"{n} signs in {time.time() - t0:.1f} s", flush=True)
