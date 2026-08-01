#!/usr/bin/env python3
import numpy as np, struct, sys, re
gt=np.load(sys.argv[1])
binf=sys.argv[2]; logf=sys.argv[3]; mapf=sys.argv[4]
int2base=np.load(mapf)
K=10
with open(binf,'rb') as f:
    ns,nq,tk=struct.unpack('III',f.read(12))
    steps=[np.frombuffer(f.read(nq*tk*4),dtype=np.int32).reshape(nq,tk) for _ in range(ns)]
S=min(ns,gt.shape[0]); recs=[]
N=len(int2base)
for s in range(S):
    pred=steps[s]; g=gt[s]; hit=0
    _nq=min(nq,pred.shape[0],g.shape[0])
    for i in range(_nq):
        pb=set()
        for x in pred[i,:K]:
            if 0<=x<N: pb.add(int(int2base[x]))
        hit+=len(pb & set(g[i,:K].tolist()))
    recs.append(100.0*hit/(_nq*K))
recs=np.array(recs)
ins=[];dele=[];sea=[]
for ln in open(logf,errors="ignore"):
    m=re.search(r"Operation \d+ \[(INSERT|SEARCH|DELETE)\] Time: [\d.]+ sec \| Throughput: ([\d.]+)",ln)
    if m:
        v=float(m.group(2))
        {"INSERT":ins,"DELETE":dele,"SEARCH":sea}[m.group(1)].append(v)
print(f"=== SVF clustered: recall@10 over {S} search steps ===")
print(f"  first={recs[0]:.2f}% final={recs[-1]:.2f}% min={recs.min():.2f}% max={recs.max():.2f}% mean={recs.mean():.2f}%")
print(f"  trajectory: " + " ".join(f"{recs[i]:.1f}" for i in range(0,S,max(1,S//12))))
print("  per-step recall@10 (all %d steps): " % S + " ".join(f"{r:.2f}" for r in recs))
import os as _os
_base=_os.path.splitext(binf)[0]
with open(_base+"_perstep.csv","w") as _cf:
    _cf.write("search_step,recall@10_pct\n")
    for i in range(S): _cf.write(f"{i},{recs[i]:.4f}\n")
print(f"  [per-step recall CSV -> {_base}_perstep.csv]")
print(f"=== throughput ===")
print(f"  insert mean={np.mean(ins):.0f}/s ({len(ins)} batches)") if ins else None
print(f"  delete mean={np.mean(dele):.0f}/s ({len(dele)} batches, bitset)") if dele else None
print(f"  search mean={np.mean(sea):.0f} QPS ({len(sea)} steps)") if sea else None
