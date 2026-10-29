"""Demo sentences from the ASL Citizen dictionary: medoid signs joined by eased SLERP transitions (PAPER_AUDIT §6.5:
interpolation, not SignSparK, for stitching). Output is ver_02's stream format, so the robot pipeline reads it as is.

Transition between sign i and i+1: N frames of per-joint SLERP from the last frame of i to the first of i+1 with
cosine easing (as SignSparK_Retrieval's stitch.blend, but on rotations, not linear rot6D). Hold frames at both ends.

    ../ver-01/.venv-signspark/bin/python dictionary/stitch_sentences.py [--render]
Output: outputs/dictionary/sentences/<slug>/{body_rot6d,hand_rot6d,hand_rot6d_left_modelframe}.npy, meta.json,
        sheet.png and video.mp4 (with --render; ver_02 renderer, EGL)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
DICT = REPO / "data" / "asl_citizen" / "dict"
OUT = REPO / "outputs" / "dictionary" / "sentences"
STREAMS = ("body_rot6d.npy", "hand_rot6d.npy", "hand_rot6d_left_modelframe.npy")
# ver_03_sr_t2p/sentences.json (approved) + ver_02's six demo sentences; ASL gloss order, ASL Citizen gloss names
SENTENCES = {
    "how_are_you": ("How are you?", ["HOW1", "YOU"]),
    "where_is_your_school": ("Where is your school?", ["YOUR", "SCHOOL", "WHERE"]),
    "please_help": ("Please help.", ["PLEASE", "HELP"]),
    "where_is_the_bathroom": ("Where is the bathroom?", ["BATHROOM", "WHERE"]),
    "sorry": ("Sorry.", ["SORRY"]),
    "see_you_tomorrow": ("See you tomorrow.", ["SEE", "YOU", "TOMORROW"]),
    "who_are_you": ("Who are you?", ["YOU", "WHO"]),
    "please_slow_down": ("Please slow down.", ["PLEASE", "SLOW"]),
    "hello_how_are_you": ("Hello, how are you?", ["HELLO", "HOW1", "YOU"]),
    "thank_you_very_much": ("Thank you very much.", ["THANKYOU"]),
    "what_is_your_name": ("What is your name?", ["YOUR", "NAME", "WHAT1"]),
    "i_am_learning_sign_language": ("I am learning sign language.", ["ME", "LEARN", "SIGNLANGUAGE"]),
    "nice_to_meet_you": ("Nice to meet you.", ["NICE", "MEET", "YOU"]),
}


def from6(x):
    d = x.reshape(len(x), -1, 6)
    a1, a2 = d[..., :3], d[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 /= np.linalg.norm(b2, axis=-1, keepdims=True)
    return np.stack([b1, b2, np.cross(b1, b2)], -2)


def transition(a, b, n):
    """Eased per-joint SLERP strictly between rot6D frames a and b (D,) -> (n, D)."""
    Ra, Rb = from6(a[None])[0], from6(b[None])[0]
    s = (1 - np.cos(np.linspace(0, 1, n + 2)[1:-1] * np.pi)) / 2
    R = np.stack([Slerp([0, 1], Rotation.from_matrix(np.stack([Ra[j], Rb[j]])))(s).as_matrix() for j in range(len(Ra))], 1)
    return R[:, :, :2, :].reshape(n, -1).astype(np.float32)


def build(glosses, index, n_trans, hold):
    parts = {k: [] for k in STREAMS}
    spans, src = [], []
    t = 0
    for i, g in enumerate(glosses):
        clip = index[g]["medoid"]
        d = DICT / "clips" / g / clip
        x = {k: np.load(d / k) for k in STREAMS}
        if i:
            for k in STREAMS:
                parts[k].append(transition(parts[k][-1][-1], x[k][0], n_trans))
            t += n_trans
        for k in STREAMS:
            parts[k].append(x[k])
        spans.append([g, t, t + len(x[STREAMS[0]])])
        src.append(f"{g}/{clip}")
        t += len(x[STREAMS[0]])
    out = {k: np.concatenate(v) for k, v in parts.items()}
    out = {k: np.concatenate([np.repeat(v[:1], hold, 0), v, np.repeat(v[-1:], hold, 0)]) for k, v in out.items()}
    return out, [[g, a + hold, b + hold] for g, a, b in spans], src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transition", type=int, default=8, help="frames at 24 fps")
    ap.add_argument("--hold", type=int, default=6)
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()
    index = json.loads((DICT / "index.json").read_text())["glosses"]
    if a.render:
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        sys.path.insert(0, str(ROOT / "ver_02"))
        import render_signspark_reference as rsr
    for slug, (english, glosses) in SENTENCES.items():
        missing = [g for g in glosses if g not in index or not index[g].get("medoid")]
        if missing:
            print(f"{slug}: missing {missing}, skipped")
            continue
        s, spans, src = build(glosses, index, a.transition, a.hold)
        d = OUT / slug
        d.mkdir(parents=True, exist_ok=True)
        for k, v in s.items():
            np.save(d / k, v)
        (d / "meta.json").write_text(json.dumps({"english": english, "glosses": glosses, "sign_spans": spans,
                                                 "clips": src, "fps": 24, "frames": len(s[STREAMS[0]]),
                                                 "stitch": f"eased SLERP, {a.transition} frames"}, indent=1))
        print(f"{slug}: {' '.join(glosses)} -> {len(s[STREAMS[0]])} frames", flush=True)
        if a.render:
            render(rsr, d, english, glosses, spans)


def render(rsr, d, english, glosses, spans):
    import cv2
    import imageio
    V, J, faces = rsr.human_mesh(d)
    R = rsr.Renderer(faces)
    gloss_at = lambda t: next((g for g, a, b in spans if a <= t < b), "(transition)")  # noqa: E731
    frames = [rsr.frame_pair(R, V, J, t, 24, f"{english}  [{gloss_at(t)}]") for t in range(len(V))]
    R.r.delete()
    w = imageio.get_writer(d / "video.mp4", fps=24, codec="libx264", pixelformat="yuv420p", quality=8)
    for f in frames:
        w.append_data(f)
    w.close()
    picks = [int((a + b) / 2) for _, a, b in spans]                       # mid-frame of every sign
    picks += [a + (b - a) // 4 for _, a, b in spans] + [a + 3 * (b - a) // 4 for _, a, b in spans]
    picks = sorted(set(picks))
    tiles = [cv2.resize(frames[t], (480, 240)) for t in picks]
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    imageio.imwrite(d / "sheet.png", np.concatenate([np.concatenate(tiles[i:i + 3], 1) for i in range(0, len(tiles), 3)], 0))


if __name__ == "__main__":
    main()
