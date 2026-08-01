#!/usr/bin/env python3
import struct, numpy as np, sys, os
AUG, DG, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
R = int(sys.argv[4]) if len(sys.argv) > 4 else 64

with open(AUG, "rb") as f:
    n, d = struct.unpack("<ii", f.read(8))
    vecs = np.fromfile(f, np.float32, n * d).reshape(n, d)
print(f"aug vectors: {vecs.shape}")

g = open(DG, "rb")
isz, = struct.unpack("<Q", g.read(8)); md, = struct.unpack("<I", g.read(4))
st,  = struct.unpack("<I", g.read(4)); nf, = struct.unpack("<Q", g.read(8))
print(f"diskann: max_degree={md} start={st} num_frozen={nf}")
deg = np.zeros(n, np.uint32); nbr = np.zeros((n, R), np.uint32); tot = 0
for i in range(n):
    GK, = struct.unpack("<I", g.read(4))
    a = np.frombuffer(g.read(4 * GK), dtype=np.uint32)
    a = a[a < n][:R]
    deg[i] = len(a); nbr[i, :len(a)] = a; tot += len(a)
g.close()
print(f"edges total={tot} avg_deg={tot/n:.2f}")

entry = d*4 + 4 + R*4
buf = np.zeros((n, entry), np.uint8)
buf[:, :d*4]      = vecs.reshape(n, -1).view(np.uint8)
buf[:, d*4:d*4+4] = deg.view(np.uint8).reshape(n, 4)
buf[:, d*4+4:]    = nbr.view(np.uint8).reshape(n, R*4)
buf.tofile(OUT)
print(f"wrote {OUT}  ({os.path.getsize(OUT)} bytes, entry={entry}, n={n})")
