#!/usr/bin/env python3
import sys, struct, os
import numpy as np
src, dst, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src, "rb") as f:
    n_hdr, dim = struct.unpack("<II", f.read(8))
actual = (os.path.getsize(src) - 8) // (dim * 4)
rows = min(N, actual)
CH = 2_000_000
with open(src, "rb") as fi, open(dst, "wb") as fo:
    fo.write(struct.pack("<II", rows, dim))
    fi.seek(8)
    left = rows
    while left > 0:
        c = min(CH, left)
        buf = np.fromfile(fi, dtype=np.float32, count=c * dim)
        if buf.size < c * dim:
            break
        buf.tofile(fo)
        left -= c
print(f"prefix {dst}: {rows} x {dim} (hdr claimed {n_hdr}, actual {actual})", flush=True)
