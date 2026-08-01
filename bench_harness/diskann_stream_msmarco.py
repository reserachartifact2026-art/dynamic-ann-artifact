#!/usr/bin/env python3
import os
import numpy as np, struct, json, time, sys, diskannpy

DATA_ROOT = os.environ["DATA_ROOT"]
BASE = os.path.join(DATA_ROOT, "freshbang_data", "msmarco_1m_base.fbin")
QUERY = os.path.join(DATA_ROOT, "freshbang", "data", "msmarco_10m", "query.fbin")
WL    = sys.argv[1]
Ls    = int(sys.argv[2]) if len(sys.argv) > 2 else 100
SEED  = 500000
K     = 10

def load(p, maxn=None):
    n, d = struct.unpack("<ii", open(p, "rb").read(8))
    if maxn: n = min(n, maxn)
    return np.fromfile(p, np.float32, n*d, offset=8).reshape(n, d), d

B, dim = load(BASE)
Q, _   = load(QUERY)
print(f"base {B.shape} query {Q.shape} dim={dim} Ls={Ls} workload={WL.split('/')[-1]}", flush=True)

idx = diskannpy.DynamicMemoryIndex(distance_metric="mips", vector_dtype=np.float32,
        max_vectors=B.shape[0]+2000, dimensions=dim, graph_degree=64, complexity=100,
        num_threads=32, initial_search_complexity=100)
idx.batch_insert(B[:SEED], np.arange(1, SEED+1, dtype=np.uint32))
print(f"seeded {SEED}", flush=True)

def ops(path):
    buf=[]; depth=0; ins=False; esc=False
    with open(path) as f:
        while True:
            c=f.read(1<<20)
            if not c: break
            for ch in c:
                buf.append(ch)
                if esc: esc=False; continue
                if ch=='\\': esc=True; continue
                if ch=='"': ins=not ins; continue
                if ins: continue
                if ch=='{': depth+=1
                elif ch=='}':
                    depth-=1
                    if depth==0:
                        yield json.loads(''.join(buf)); buf=[]

pending=0; recs=[]; ni=nd=0; t0=time.perf_counter()
for o in ops(WL):
    t=o["op_type"]
    if t=="insert":
        ids=np.asarray(o["vector_ids"],dtype=np.int64)
        idx.batch_insert(B[ids], (ids+1).astype(np.uint32)); ni+=len(ids)
    elif t=="delete":
        ids=np.asarray(o["vector_ids"],dtype=np.int64)
        for i in ids: idx.mark_deleted(int(i)+1)
        pending+=len(ids); nd+=len(ids)
        if pending>=100000:
            idx.consolidate_delete(); pending=0
    elif t=="search":
        gt=np.asarray([r[:K] for r in o["groundtruth"]],dtype=np.int64)
        nq=gt.shape[0]
        tags,_=idx.batch_search(Q[:nq],K,Ls,32)
        pred=tags.astype(np.int64)-1
        hit=sum(len(set(pred[i].tolist()) & set(gt[i].tolist())) for i in range(nq))
        recs.append(100.0*hit/(nq*K))
        if len(recs)%20==0: print(f"  step {len(recs)}: recall={recs[-1]:.2f}%", flush=True)

print(f"\nRESULT {WL.split('/')[-1]}  Ls={Ls}  inserts={ni} deletes={nd}", flush=True)
print(f"  steps={len(recs)} MEAN={sum(recs)/len(recs):.2f}%  min={min(recs):.2f} max={max(recs):.2f}  wall={time.perf_counter()-t0:.0f}s", flush=True)
print("  trajectory:", " ".join(f"{r:.0f}" for r in recs[::max(1,len(recs)//20)]), flush=True)
