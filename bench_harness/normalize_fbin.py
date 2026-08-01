#!/usr/bin/env python3
import sys, numpy as np

inp, outp = sys.argv[1], sys.argv[2]
nrows = int(sys.argv[3]) if len(sys.argv) > 3 else 0

with open(inp, "rb") as f:
    n, d = np.frombuffer(f.read(8), dtype=np.uint32)
n, d = int(n), int(d)
if nrows and nrows < n: n = nrows
print(f"normalizing {n} rows x {d} dims from {inp}")

CH = 200_000
with open(inp, "rb") as fin, open(outp, "wb") as fout:
    fin.read(8)
    fout.write(np.array([n, d], dtype=np.uint32).tobytes())
    done = 0
    while done < n:
        m = min(CH, n - done)
        buf = np.frombuffer(fin.read(m * d * 4), dtype=np.float32).reshape(m, d).copy()
        norms = np.linalg.norm(buf, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        (buf / norms).astype(np.float32).tofile(fout)
        done += m
        if done % 1_000_000 == 0: print(f"  {done}/{n}")
print("wrote", outp)
