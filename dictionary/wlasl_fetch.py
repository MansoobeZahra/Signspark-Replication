"""Fetch a few WLASL v0.3 videos per gloss for the ASL-dictionary prototype (user-approved, 2026-10-26).

Per gloss: instances of variation 0 (WLASL's main form of the sign), direct-file sources first (one isolated sign
per file), then YouTube sections via yt-dlp; stop after --per-gloss clips that ffprobe can read. Each clip is
cut to WLASL's [frame_start, frame_end] (1-based, -1 = end) and re-encoded to mp4. Downloads are data only:
nothing in them is executed.

    .venv-sltk/bin/python dictionary/wlasl_fetch.py [--glosses how you ...] [--per-gloss 3]
Output: data/wlasl/videos/<gloss>/<video_id>.mp4 + data/wlasl/fetch_log.json
"""
import argparse
import json
import subprocess
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
META = REPO.parent / "ver_03_sr_t2p" / "data" / "meta" / "WLASL_v0.3.json"
OUT = REPO / "data" / "wlasl"
# ver_03_sr_t2p/sentences.json (13 glosses) + the ver_02 demo sentences' glosses
GLOSSES = ["how", "you", "your", "school", "where", "please", "help", "bathroom", "sorry", "see", "tomorrow", "who",
           "slow", "hello", "me", "good", "need", "nice", "learn", "language", "meet", "thank you", "name"]
UA = "Mozilla/5.0 (X11; Linux x86_64)"


def probe_frames(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                        "stream=nb_read_packets", "-of", "csv=p=0", str(p)], capture_output=True, text=True)
    try:
        return int(r.stdout.strip().split(",")[0])
    except ValueError:
        return 0


def fetch(inst, raw):
    url = inst["url"]
    if "youtu" in url:
        fps = inst.get("fps", 25)
        a, b = (inst["frame_start"] - 1) / fps, (inst["frame_end"] / fps if inst["frame_end"] > 0 else None)
        sec = f"*{a:.2f}-{b:.2f}" if b else f"*{a:.2f}-inf"
        r = subprocess.run([str(REPO / ".venv-sltk" / "bin" / "yt-dlp"), "-q", "--no-playlist", "-f", "mp4/best",
                            "--download-sections", sec, "-o", str(raw), url], capture_output=True, text=True, timeout=180)
        return raw.exists(), r.stderr[-300:]
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as f:
        data = f.read(50_000_000)
    raw.write_bytes(data)
    return True, ""


def cut(inst, raw, out):
    """Re-encode to mp4 at the source fps; frame range only for direct files (yt-dlp already cut the section)."""
    vf = []
    if "youtu" not in inst["url"] and inst["frame_end"] > 0:
        vf = ["-vf", f"select='between(n\\,{inst['frame_start'] - 1}\\,{inst['frame_end'] - 1})',setpts=N/FRAME_RATE/TB"]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), *vf, "-an", "-c:v", "libx264", "-pix_fmt",
                    "yuv420p", str(out)], capture_output=True, timeout=120)
    return probe_frames(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glosses", nargs="+", default=GLOSSES)
    ap.add_argument("--per-gloss", type=int, default=3)
    a = ap.parse_args()
    meta = {e["gloss"]: e["instances"] for e in json.loads(META.read_text())}
    log_p = OUT / "fetch_log.json"
    log = json.loads(log_p.read_text()) if log_p.exists() else {}
    for g in a.glosses:
        d = OUT / "videos" / g.replace(" ", "_")
        d.mkdir(parents=True, exist_ok=True)
        got = [p.stem for p in d.glob("*.mp4")]
        cands = sorted(meta[g], key=lambda i: (i["variation_id"] != 0, "youtu" in i["url"], i["split"] != "train"))
        for inst in cands:
            if len(got) >= a.per_gloss:
                break
            vid = inst["video_id"]
            if vid in got or log.get(vid, {}).get("ok") is False:
                continue
            raw = OUT / "raw" / f"{vid}.src"
            raw.parent.mkdir(parents=True, exist_ok=True)
            entry = {"gloss": g, "url": inst["url"], "source": inst["source"], "signer": inst["signer_id"],
                     "variation": inst["variation_id"], "frames": [inst["frame_start"], inst["frame_end"]],
                     "fps": inst.get("fps"), "bbox": inst["bbox"]}
            try:
                ok, err = fetch(inst, raw)
                n = cut(inst, raw, d / f"{vid}.mp4") if ok else 0
                entry.update(ok=n > 5, n_frames=n, err=err)
            except Exception as e:  # dead links are the norm for WLASL
                entry.update(ok=False, err=f"{type(e).__name__}: {e}"[:300])
            if not entry["ok"]:
                (d / f"{vid}.mp4").unlink(missing_ok=True)
            else:
                got.append(vid)
            raw.unlink(missing_ok=True)
            log[vid] = entry
            log_p.write_text(json.dumps(log, indent=1))
        print(f"{g:10} {len(got)} clips: {got}", flush=True)


if __name__ == "__main__":
    main()
