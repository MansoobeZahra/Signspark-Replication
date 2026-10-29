"""Extract the How2Sign TEST clips from Neural-Sign-Actors.zip (SOKE's How2Sign fittings) into the layout MaDiS's
loader reads: data/How2Sign/poses_video_level/test/poses/<SENTENCE_NAME>/<SENTENCE_NAME>_<frame>_3D.pkl.

Works on a partially downloaded archive: Dropbox streams the folder as a zip whose central directory is only at
the end, so entries are read from their local headers in order; each entry is raw deflate (self-terminating),
decompressed until the stream ends (sizes may be in trailing data descriptors). Only entries whose clip is in
SOKE's test split CSV are written. Completeness is checked against the frame indices (no gaps, starts at 0).

    python3 extract_nsa_test.py
"""

import csv
import re
import struct
import zlib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ZIP = ROOT / "data" / "Neural-Sign-Actors.zip"
OUT = ROOT / "data" / "How2Sign" / "poses_video_level" / "test" / "poses"
CSV = ROOT / "data" / "soke_splits" / "how2sign_realigned_test_preprocessed_fps.csv"
STOP = 800_000_000      # all test entries end before byte 0.773e9 (checked on the partial download)


def main():
    test = {r["SENTENCE_NAME"] for r in csv.DictReader(open(CSV))}
    raw = open(ZIP, "rb").read(STOP + 50_000_000)
    b = memoryview(raw)                                  # zero-copy slices
    n_written, frames = 0, defaultdict(list)
    for m in re.finditer(rb"PK\x03\x04", raw[:STOP]):
        o = m.start()
        try:
            method, = struct.unpack("<H", b[o + 8:o + 10])
            n, e = struct.unpack("<HH", b[o + 26:o + 30])
            name = bytes(b[o + 30:o + 30 + n]).decode()
        except (struct.error, UnicodeDecodeError):
            continue
        p = name.split("/")
        if not (len(p) == 3 and p[0] == "poses" and p[1] in test and p[2].endswith("_3D.pkl")):
            continue
        start = o + 30 + n + e
        if method == 8:
            d = zlib.decompressobj(-15)
            data = d.decompress(b[start:start + 5_000_000])
            assert d.eof, name
        else:
            size, = struct.unpack("<I", b[o + 18:o + 22])
            data = bytes(b[start:start + size])
        dst = OUT / p[1] / p[2]
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        frames[p[1]].append(int(p[2].rsplit("_", 2)[-2]))
        n_written += 1
    complete = sum(1 for k in test if frames.get(k) and sorted(frames[k]) == list(range(len(frames[k]))))
    print(f"wrote {n_written} frame files for {len(frames)}/{len(test)} test clips; complete (0..N-1, no gaps): {complete}")


if __name__ == "__main__":
    main()
