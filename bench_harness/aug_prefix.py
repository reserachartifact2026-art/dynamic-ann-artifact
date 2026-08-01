#!/usr/bin/env python3
import struct, numpy as np, sys
RAW, OUT = sys.argv[1], sys.argv[2]
N_seed, N_M = int(sys.argv[3]), int(sys.argv[4])
with open(RAW, "rb") as f:
    n, d = struct.unpack("<ii", f.read(8))
N_M = min(N_M, n); N_seed = min(N_seed, n)
CH = 500000; off = 8
M2 = 0.0; rd = 0
while rd < N_M:
    k = min(CH, N_M - rd)
    a = np.fromfile(RAW, np.float32, k*d, offset=off + rd*d*4).reshape(k, d)
    M2 = max(M2, float((a*a).sum(1).max())); rd += k
M = float(np.sqrt(M2))
print(f"M over first {N_M} = {M:.6f}; augmenting first {N_seed} rows (d={d}->{d+1}) -> {OUT}", flush=True)
with open(OUT, "wb") as o:
    o.write(struct.pack("<ii", N_seed, d+1))
    rd = 0
    while rd < N_seed:
        k = min(CH, N_seed - rd)
        a = np.fromfile(RAW, np.float32, k*d, offset=off + rd*d*4).reshape(k, d)
        s = (a*a).sum(1)
        extra = np.sqrt(np.maximum(M*M - s, 0)).astype(np.float32)
        np.concatenate([a, extra[:, None]], 1).astype(np.float32).tofile(o); rd += k
print("wrote", OUT, "M=", M, flush=True)
