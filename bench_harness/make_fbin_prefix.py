import struct, sys, os

src = sys.argv[1]
dst = sys.argv[2]
N   = int(sys.argv[3])

with open(src, "rb") as f:
    n, d = struct.unpack("<ii", f.read(8))
    assert N <= n, f"{N} > {n}"
    nbytes = N * d * 4
    with open(dst, "wb") as g:
        g.write(struct.pack("<ii", N, d))
        remaining = nbytes
        while remaining > 0:
            chunk = f.read(min(1 << 24, remaining))
            if not chunk:
                break
            g.write(chunk)
            remaining -= len(chunk)
print(f"wrote {dst}: {N} x {d} ({os.path.getsize(dst)} bytes, expect {8+nbytes})")
