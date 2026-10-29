"""Side-by-side renders for steps 2-3 (GT | candidates), through model_tests/render_side_by_side.py's mesh
conversion and renderer (imported unchanged; EGL renderer when available, else its CPU fallback).

    .venv-signspark/bin/python render_bestofn.py                    # step 2: renders/bestofn/<clip>/sheet.png
    .venv-signspark/bin/python render_bestofn.py --video G1GUMky8kWc_20-2-rgb_front
"""

import argparse
import json
import sys

import cv2
import imageio
import numpy as np

import bt_common as bc

sys.path.insert(0, str(bc.ROOT.parent / "model_tests"))
import render_side_by_side as rs  # noqa: E402

OUTD = bc.ROOT / "outputs" / "bestofn"
RD = bc.ROOT / "renders" / "bestofn"


def panel(f240, label):
    """240-d features (left hand true SMPL-X-left) -> (label, V, J)."""
    V, J, faces = rs.mesh(f240[:, :60], f240[:, 60:150], f240[:, 150:], False)
    rs.FACES[0] = faces
    return label, V, J


def render(out, text, panels, fps, video):
    out.mkdir(parents=True, exist_ok=True)
    try:
        R = rs.rsr.Renderer(rs.FACES[0])
        pair = lambda V, J, k: rs.rsr.frame_pair(R, V, J, k, fps, text)  # noqa: E731
    except Exception as e:
        print(f"EGL renderer unavailable ({type(e).__name__}); CPU fallback", flush=True)
        R, soft = None, rs.SoftRenderer(rs.FACES[0])
        pair = lambda V, J, k: soft.frame_pair(V, J, k, fps, text)  # noqa: E731
    T = max(len(V) for _, V, _ in panels)

    def frame(t):
        rows = []
        for lab, V, J in panels:
            img = pair(V, J, min(t, len(V) - 1))
            bar = np.full((34, img.shape[1], 3), 45, np.uint8)
            rs.rsr.header(bar, [lab[:110]])
            rows.append(np.concatenate([bar, img], 0))
        return np.concatenate(rows, 0)

    picks = np.linspace(0, T - 1, 6).astype(int)
    tiles = [cv2.resize(frame(t), (480, 274 * len(panels))) for t in picks]
    imageio.imwrite(out / "sheet.png", np.concatenate(tiles, 1))
    if video:
        step = 1 if R is not None else rs.VIDEO_STEP
        w = imageio.get_writer(out / "video.mp4", fps=fps / step, codec="libx264", pixelformat="yuv420p", quality=8)
        for t in range(0, T, step):
            w.append_data(frame(t))
        w.close()
    print(f"wrote {out.relative_to(bc.ROOT)} ({T} frames)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", nargs="*", default=[], help="clip ids (or demoK) that also get a video")
    a = ap.parse_args()
    from bt_bestofn import TEST_IDX, sentences
    sub, test, demo, _ = sentences()
    Z = np.load(OUTD / "cands.npz", allow_pickle=True)
    C = np.load(OUTD / "ctrl_wrongtext.npz", allow_pickle=True)
    hyps, hc = np.load(OUTD / "hyps.npy"), np.load(OUTD / "ctrl_hyps.npy")
    CH = np.load(OUTD / "chrf_matrix.npy")
    res = json.loads((OUTD / "results.json").read_text())
    nt, NC = len(test), Z["seeds"].shape[1]
    ex = {e["id"]: e for e in res["examples"]}

    # test clips: the counting clip, the clip with the highest best-of-32 chrF, and test sentence 0
    best = max(range(nt), key=lambda i: CH[i, :, i].max())
    for i in dict.fromkeys([nt - 1, best, 0]):
        c, e = test[i], ex[test[i]["id"]]
        cc = np.array([bc.sent_chrf(e["ref"], h) for h in hc[i]])
        j32, jc = e["best32_index"], int(cc.argmax())
        panels = [panel(bc.gt_feats(c), "REAL recording (LMDB)"),
                  panel(Z["feats"][i * NC].astype(np.float32), f"N=1 seed {Z['seeds'][i, 0]} | BT: {hyps[i, 0]}"),
                  panel(Z["feats"][i * NC + j32].astype(np.float32),
                        f"BT best-of-32 (seed {Z['seeds'][i, j32]}) | BT: {hyps[i, j32]}"),
                  panel(C["feats"][i * NC + jc].astype(np.float32),
                        f"CONTROL wrong text, BT best-of-32 | BT: {hc[i, jc]}")]
        render(RD / c["id"], c["text"], panels, 20, c["id"] in a.video)
    for d in (0, 5):
        k = nt + d
        r = res["demo"][d]
        j = r["best32_index"]
        panels = [panel(Z["feats"][k * NC].astype(np.float32), f"N=1 | BT: {r['hyp_n1']}"),
                  panel(Z["feats"][k * NC + j].astype(np.float32), f"BT best-of-32 (seed {Z['seeds'][k, j]}) | BT: {r['hyp_best32']}")]
        render(RD / f"demo{d}", r["text"], panels, 20, f"demo{d}" in a.video)


if __name__ == "__main__":
    main()
