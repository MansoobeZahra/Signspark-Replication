"""NLF SMPL-X fits for the WLASL prototype clips (CPU), then SignSparK rot6D streams + LMDB records.

Per clip: YOLOv8x (sltk's nlf/yolov8x.pt) on 8 evenly spaced frames -> highest-confidence person box per frame
-> one median box for the clip (isolated signs: one static signer); NLF-L (checkpoints/nlf, fp32 CPU patch, see
PATCH.md) estimate_parametric_batched(model_name="smplx") on every frame with that box.

SMPL-X pose (55 x 3 axis-angle): 0 root, 1-21 body, 22 jaw, 23-24 eyes, 25-39 left hand, 40-54 right hand.
Streams (DATA.md §2 and ver_02's format): body_features = rot6D (first two ROWS) of joints 1-21 (126; SignSparK
keeps [:, 66:] = joints 12-21), left/right_features = 15 x rot6D, left hand in true SMPL-X-left convention
(hand_rot6d_left_modelframe.npy is the flipped copy ver_02 expects). NLF's hand poses are absolute rotations
(smplfitter's SMPL-X has no hand-mean offset), i.e. the flat_hand_mean=True convention SignSparK renders with.
Hands here are NLF's, not WiLoR's: WiLoR needs MANO_RIGHT.pkl (license-gated) and replaces them once available.

    CUDA_VISIBLE_DEVICES= .venv-sltk/bin/python dictionary/extract_nlf.py [--threads 8]
Output: data/wlasl/nlf/<gloss>/<video_id>.npz (raw fit) and data/wlasl/streams/<gloss>/<video_id>/*.npy
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision  # noqa: F401  (registers the torchvision ops the TorchScript model calls)
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
W = REPO / "data" / "wlasl"
NLF = REPO / "checkpoints" / "nlf" / "nlf_l_multi_0.3.2_fp32_v2.torchscript"
YOLO_W = Path.home() / ".cache" / "sltk" / "weights" / "nlf" / "yolov8x.pt"
FLIP = np.diag([1.0, -1.0, -1.0])


def read_frames(p):
    cap = cv2.VideoCapture(str(p))
    fps, fr = cap.get(cv2.CAP_PROP_FPS), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f[..., ::-1])
    return np.stack(fr), fps


def clip_box(det, rgb):
    idx = np.linspace(0, len(rgb) - 1, min(8, len(rgb))).astype(int)
    res = det([rgb[i][..., ::-1] for i in idx], classes=[0], verbose=False, device="cpu")
    bx = [r.boxes.xyxy[r.boxes.conf.argmax()].numpy() for r in res if len(r.boxes)]
    if not bx:
        return None
    x1, y1, x2, y2 = np.median(bx, 0)
    return np.array([x1, y1, x2 - x1, y2 - y1], np.float32)  # NLF boxes are xywh


def rot6d(aa):
    """(T, J, 3) axis-angle -> (T, J*6), first two rows of each rotation matrix (SignSparK convention)."""
    T, J, _ = aa.shape
    m = Rotation.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(T, J, 3, 3)
    return m[:, :, :2, :].reshape(T, J * 6).astype(np.float32)


def to_streams(pose):
    p = pose.reshape(len(pose), 55, 3)
    lh = rot6d(p[:, 25:40])
    R = Rotation.from_rotvec(p[:, 25:40].reshape(-1, 3)).as_matrix()
    lh_model = (FLIP @ R @ FLIP)[:, :2, :].reshape(len(p), 90).astype(np.float32)
    return {"body_features": rot6d(p[:, 1:22]), "right_features": rot6d(p[:, 40:55]), "left_features": lh,
            "left_model": lh_model, "jaw": rot6d(p[:, 22:23])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from ultralytics import YOLO
    det = YOLO(str(YOLO_W))
    m = torch.jit.load(str(NLF), map_location="cpu").float().eval()
    log = {}
    for vid_p in sorted((W / "videos").glob("*/*.mp4")):
        gloss, vid = vid_p.parent.name, vid_p.stem
        out = W / "nlf" / gloss / f"{vid}.npz"
        if out.exists():
            continue
        rgb, fps = read_frames(vid_p)
        box = clip_box(det, rgb)
        if box is None:
            log[vid] = "no person"
            continue
        keys = ("pose", "betas", "trans", "joints3d")
        acc = {k: [] for k in keys}
        with torch.inference_mode():
            for i in range(0, len(rgb), a.batch):
                v = torch.from_numpy(rgb[i:i + a.batch].copy()).permute(0, 3, 1, 2)
                o = m.estimate_parametric_batched(v, [torch.from_numpy(box[None])] * len(v), model_name="smplx")
                for k in keys:  # one box per frame -> first (only) person
                    acc[k] += [x[0].numpy() if len(x) else np.full(o[k][0].shape[1:], np.nan) for x in o[k]]
        fit = {k: np.stack(v).astype(np.float32) for k, v in acc.items()}
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **fit, fps=fps, box_xywh=box)
        s = to_streams(fit["pose"])
        sd = W / "streams" / gloss / vid
        sd.mkdir(parents=True, exist_ok=True)
        np.save(sd / "body_rot6d.npy", s["body_features"][:, 66:])
        np.save(sd / "hand_rot6d.npy", s["right_features"])
        np.save(sd / "hand_rot6d_left_modelframe.npy", s["left_model"])
        (sd / "meta.json").write_text(json.dumps({"gloss": gloss, "video_id": vid, "fps": fps, "frames": len(rgb),
                                                  "source": "WLASL v0.3 video, NLF-L SMPL-X (hands: NLF)"}))
        log[vid] = len(rgb)
        print(f"{gloss:10} {vid} {len(rgb)} frames", flush=True)
    (W / "nlf_log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
