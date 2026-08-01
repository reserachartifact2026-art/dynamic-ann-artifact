#!/usr/bin/env python3
import os
import struct, numpy as np
DATA_ROOT = os.environ["DATA_ROOT"]
PROJECT_ROOT = os.environ["PROJECT_ROOT"]
BASE = os.path.join(DATA_ROOT, "freshbang", "data", "msmarco_10m", "base.fbin")
OUT = os.path.join(PROJECT_ROOT, "bench_harness", "prefix", "msmarco_5m_aug769.fbin")
N = 5000000
with open(BASE, "rb") as f:
    n, d = struct.unpack("<ii", f.read(8))
print(f"base n={n} d={d}; computing M over all (streamed)...")
M2 = 0.0; CH = 500000; off = 8
rd = 0
while rd < n:
    k = min(CH, n - rd)
    a = np.fromfile(BASE, np.float32, k*d, offset=off + rd*d*4).reshape(k, d)
    M2 = max(M2, float((a*a).sum(1).max())); rd += k
M = float(np.sqrt(M2))
print(f"M={M:.4f}; augmenting first {N} rows -> {OUT}")
with open(OUT, "wb") as o:
    o.write(struct.pack("<ii", N, d+1))
    rd = 0
    while rd < N:
        k = min(CH, N - rd)
        a = np.fromfile(BASE, np.float32, k*d, offset=off + rd*d*4).reshape(k, d)
        s = (a*a).sum(1)
        extra = np.sqrt(np.maximum(M*M - s, 0)).astype(np.float32)
        np.concatenate([a, extra[:, None]], 1).astype(np.float32).tofile(o)
        rd += k
print("wrote", OUT)
