#!/usr/bin/env python3
import numpy as np, struct, sys, re, os

gt = np.load(sys.argv[1])
binf = sys.argv[2]
logf = sys.argv[3]
K = 10

with open(binf,'rb') as f:
    ns,nq,tk = struct.unpack('III', f.read(12))
    steps = [np.frombuffer(f.read(nq*tk*4), dtype=np.int32).reshape(nq,tk) for _ in range(ns)]

S = min(ns, gt.shape[0])
recs=[]
for s in range(S):
    pred=steps[s]; g=gt[s]
    hit=0
    _nq = min(nq, pred.shape[0], g.shape[0])
    for i in range(_nq):
        hit += len(set(pred[i,:K].tolist()) & set(g[i,:K].tolist()))
    recs.append(hit/(_nq*K))
recs=np.array(recs)

ops=[]
for ln in open(logf, errors='ignore'):
    m=re.search(r"Operation (\d+) \[(INSERT|SEARCH|DELETE)\] Time: ([\d.]+) sec \| Throughput: ([\d.]+) (\S+)", ln)
    if m:
        ops.append((int(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4)), m.group(5)))

ins=[o[3] for o in ops if o[1]=="INSERT"]
dele=[o[3] for o in ops if o[1]=="DELETE"]
sea=[o[3] for o in ops if o[1]=="SEARCH"]
times={ "INSERT":sum(o[2] for o in ops if o[1]=="INSERT"),
        "DELETE":sum(o[2] for o in ops if o[1]=="DELETE"),
        "SEARCH":sum(o[2] for o in ops if o[1]=="SEARCH") }

print(f"=== RECALL@10 over {S} search steps ===")
print(f"  build/first: {recs[0]*100:.2f}%   final: {recs[-1]*100:.2f}%")
print(f"  min={recs.min()*100:.2f}%  max={recs.max()*100:.2f}%  mean={recs.mean()*100:.2f}%")
print(f"  trajectory (every ~{max(1,S//10)}): " + " ".join(f"{recs[i]*100:.1f}" for i in range(0,S,max(1,S//10))))
print("  per-step recall@10 (all %d steps): " % S + " ".join(f"{r*100:.2f}" for r in recs))
_base = os.path.splitext(binf)[0]
_qs = [o[3] for o in ops if o[1]=="SEARCH"]
with open(_base+"_perstep.csv","w") as _cf:
    _cf.write("search_step,recall@10_pct,search_qps\n")
    for i in range(S):
        _q = f"{_qs[i]:.0f}" if i < len(_qs) else ""
        _cf.write(f"{i},{recs[i]*100:.4f},{_q}\n")
print(f"  [per-step recall+qps CSV -> {_base}_perstep.csv]")
with open(_base+"_ops.csv","w") as _cf:
    _cf.write("op_num,type,time_s,throughput,unit\n")
    for o in ops:
        _cf.write(f"{o[0]},{o[1]},{o[2]},{o[3]},{o[4]}\n")
print(f"  [per-op throughput CSV -> {_base}_ops.csv]")
print(f"=== THROUGHPUT ===")
print(f"  INSERT batches={len(ins)} mean={np.mean(ins):.0f} vec/s" if ins else "  no inserts")
print(f"  DELETE batches={len(dele)} mean={np.mean(dele):.0f} vec/s (bitset mark)" if dele else "  no deletes")
print(f"  SEARCH steps={len(sea)} mean={np.mean(sea):.0f} QPS  (min {np.min(sea):.0f}, max {np.max(sea):.0f})" if sea else "  no searches")
print(f"=== WALL (op time only) === insert={times['INSERT']:.1f}s delete={times['DELETE']:.3f}s search={times['SEARCH']:.1f}s")
