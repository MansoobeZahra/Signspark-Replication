"""ASL Citizen (or any folder of isolated-sign videos) -> sltk WiLoR + NLF .h5 features, models loaded once (GPU).

The authors' extraction stack (signlangtk 0.3.1, as in DATA.md §3 / upstream/fast): WiLoR (hands, MANO; needs
MANO_RIGHT.pkl in ~/.cache/sltk/weights/body_models/mano) and NLF-L (body, SMPL-X). The `smplfx` re-optimisation
is not run (it needs TEASER/FLAME + RTMPose). FAST runs afterwards on the WiLoR folder:
    .venv-sltk/bin/python upstream/fast/segment.py data/asl_citizen/feats/wilor -o data/asl_citizen/feats/fast.pt

    .venv-sltk/bin/python dictionary/aslc_extract.py [--glosses HELLO HOW1 ...] [--device cuda:0]
Output: data/asl_citizen/feats/{wilor,nlf}/<video stem>.h5, extract_log.json
"""
import argparse
import csv
import json
import time
from pathlib import Path

import sys

import torch
from sltk.extraction.nlf import NLFConfig, NLFExtractor

import sltk.extraction.wilor as _sw
from sltk.extraction.wilor import WiLoRConfig, WiLoRExtractor
from sltk.extraction.wilor_deps import chumpy_stub  # noqa: F401  (as WiLoRExtractor.load_model does)

sys.path.insert(0, str(Path(_sw.__file__).parent / "wilor_deps"))   # sltk imports it as top-level `wilor_processor`
import wilor_processor as _wp  # noqa: E402

# sltk 0.3.1 bug: _letterbox_gpu's bilinear resize leaves white pixels at 1.0000002; ultralytics then treats the
# [0,1] batch as [0,255] and divides by 255 again ("max value is 1.0. Dividing input by 255"), so the hand detector
# sees a black image. That silently dropped every hand in 44% of ASL Citizen clips (bright backgrounds). Clamp it.
_letterbox = _wp.WiLORProcessorOptimized._letterbox_gpu


def _letterbox_clamped(self, *a, **k):
    out = _letterbox(self, *a, **k)
    x = out[0].clamp_(0.0, 1.0)
    if x.is_cuda:  # the detector runs on its own stream without waiting for this one
        torch.cuda.current_stream(x.device).synchronize()
    return (x, *out[1:])


_wp.WiLORProcessorOptimized._letterbox_gpu = _letterbox_clamped

REPO = Path(__file__).resolve().parents[1]
AC = REPO / "data" / "asl_citizen" / "ASL_Citizen"
FEATS = REPO / "data" / "asl_citizen" / "feats"


def videos(glosses):
    out = []
    for s in ("train", "val", "test"):
        for r in csv.DictReader(open(AC / "splits" / f"{s}.csv")):
            if (glosses is None or r["Gloss"] in glosses) and (AC / "videos" / r["Video file"]).exists():
                out.append(AC / "videos" / r["Video file"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glosses", nargs="*", default=None, help="default: every extracted video")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--backends", nargs="+", default=["wilor", "nlf"], choices=["wilor", "nlf"])
    a = ap.parse_args()
    vids = videos(set(a.glosses) if a.glosses else None)
    log_p = FEATS / "extract_log.json"
    log = json.loads(log_p.read_text()) if log_p.exists() else {}
    make = {"wilor": lambda: WiLoRExtractor(WiLoRConfig(device=a.device)),
            "nlf": lambda: NLFExtractor(NLFConfig(device=a.device))}
    for name in a.backends:
        ext = make[name]()
        d = FEATS / name
        d.mkdir(parents=True, exist_ok=True)
        t0, n = time.time(), 0
        for v in vids:
            out = d / f"{v.stem}.h5"
            if out.exists():
                continue
            try:
                ext.extract_from_video(str(v), str(out))
                n += 1
            except Exception as e:  # keep going; a bad video must not stop the batch
                log[f"{name}/{v.stem}"] = f"{type(e).__name__}: {e}"[:300]
                out.unlink(missing_ok=True)
            if n and n % 50 == 0:
                print(f"{name}: {n} videos, {(time.time() - t0) / n:.1f} s/video", flush=True)
                log_p.write_text(json.dumps(log, indent=1))
        print(f"{name} done: {n} new videos in {time.time() - t0:.0f} s", flush=True)
        del ext
    log_p.write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
