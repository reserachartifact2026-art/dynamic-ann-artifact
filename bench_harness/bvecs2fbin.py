#!/usr/bin/env python3
import sys, struct, os
import numpy as np
src, dst, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src, "rb") as f:
    dim = struct.unpack("<i", f.read(4))[0]
rec = 4 + dim
actual = os.path.getsize(src) // rec
rows = min(N, actual)
CH = 1_000_000
with open(src, "rb") as fi, open(dst, "wb") as fo:
    fo.write(struct.pack("<II", rows, dim))
    left = rows
    while left > 0:
        c = min(CH, left)
        raw = np.frombuffer(fi.read(c * rec), dtype=np.uint8).reshape(c, rec)
        raw[:, 4:].astype(np.float32).tofile(fo)
        left -= c
        print(f"  {rows-left}/{rows}", flush=True)
print(f"converted {dst}: {rows} x {dim} (bvecs actual {actual})", flush=True)
