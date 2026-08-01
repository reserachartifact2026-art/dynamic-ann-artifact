#!/usr/bin/env python3
import sys, re, csv, os, struct
import numpy as np

wl, fda_log, fda2_log, svf_log, svf_bin, svf_gt, outdir = sys.argv[1:8]
os.makedirs(outdir, exist_ok=True)
K=10

def parse_fda(path):
    if not path or not os.path.exists(path): return None
    steps=[]
    recall=[]
    in_tbl=False
    for ln in open(path, errors="ignore"):
        m=re.search(r"\[PSTEP\] op=(\w+) idx=(\d+) n=(\d+) ms=([\d.]+) thr=([\d.]+)", ln)
        if m:
            steps.append((m.group(1), int(m.group(3)), float(m.group(4)), float(m.group(5))))
            continue
        if "Per-batch recall" in ln: in_tbl=True; continue
        if in_tbl:
            mm=re.match(r"\s*(\d+)\s*\|\s*([\d.]+)%", ln)
            if mm: recall.append(float(mm.group(2)))
            elif ln.strip()=="" or ln.startswith("==="): in_tbl=False
    search_recall=iter(recall)
    out=[]
    for (op,n,ms,thr) in steps:
        r = next(search_recall) if op=="search" else None
        out.append((op,n,ms,thr,r))
    return out

def parse_svf(logp, binp, gtp):
    if not (logp and os.path.exists(logp)): return None
    ops=[]
    for ln in open(logp, errors="ignore"):
        m=re.search(r"Operation (\d+) \[(INSERT|SEARCH|DELETE)\] Time: ([\d.]+) sec \| Throughput: ([\d.]+)", ln)
        if m: ops.append((m.group(2).lower(), float(m.group(3))*1000.0, float(m.group(4))))
    recall=[]
    if binp and os.path.exists(binp) and gtp and os.path.exists(gtp):
        gt=np.load(gtp)
        with open(binp,'rb') as f:
            ns,nq,tk=struct.unpack('III',f.read(12))
            stepsb=[np.frombuffer(f.read(nq*tk*4),dtype=np.int32).reshape(nq,tk) for _ in range(ns)]
        for s in range(min(ns,gt.shape[0])):
            hit=sum(len(set(stepsb[s][i,:K])&set(gt[s][i,:K])) for i in range(nq))
            recall.append(100.0*hit/(nq*K))
    sr=iter(recall); out=[]
    for (op,ms,thr) in ops:
        r=next(sr,None) if op=="search" else None
        out.append((op,None,ms,thr,r))
    return out

def write_csv(name, rows):
    if rows is None: return
    p=os.path.join(outdir,f"{wl}_{name}_perstep.csv")
    with open(p,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["step","op","count","ms","throughput","recall@10"])
        for i,(op,n,ms,thr,r) in enumerate(rows):
            w.writerow([i,op,n if n else "",f"{ms:.3f}",f"{thr:.1f}",f"{r:.2f}" if r is not None else ""])
    return p

def summary(name, rows):
    if rows is None: print(f"  {name}: (no data)"); return None
    ins=[r[3] for r in rows if r[0] in ("insert",)]
    dele=[r[3] for r in rows if r[0] in ("delete",)]
    sea_q=[r[3] for r in rows if r[0]=="search"]
    rec=[r[4] for r in rows if r[0]=="search" and r[4] is not None]
    def mn(x): return sum(x)/len(x) if x else 0
    print(f"  {name:5s}: steps={len(rows)} | ins~{mn(ins):.0f}/s | del~{mn(dele):.0f}/s | search~{mn(sea_q):.0f}qps | recall mean={mn(rec):.2f}% ({rec[0] if rec else 0:.2f}->{rec[-1] if rec else 0:.2f})")
    return rec, sea_q

fda=parse_fda(fda_log); fda2=parse_fda(fda2_log); svf=parse_svf(svf_log,svf_bin,svf_gt)
print(f"\n######## {wl} ########")
for nm,rows in (("fda",fda),("fda2",fda2),("svf",svf)):
    write_csv(nm,rows)
print("=== summary ===")
rf=summary("fda",fda); rf2=summary("fda2",fda2); rs=summary("svf",svf)

def srec(x): return x[0] if x else []
def sqps(x): return x[1] if x else []
recs={"fda":srec(rf),"fda2":srec(rf2),"svf":srec(rs)}
qps ={"fda":sqps(rf),"fda2":sqps(rf2),"svf":sqps(rs)}
n=max((len(v) for v in recs.values()), default=0)
print(f"\n=== per-search-step recall@10  (workload={wl}) ===")
print("step | fda2_R  fda2_QPS | fda_R   fda_QPS | svf_R   svf_QPS")
for i in range(n):
    def g(d,sys_,fmt):
        v=d[sys_]; return (fmt%v[i]) if i<len(v) else "  -   "
    print(f"{i:4d} | {g(recs,'fda2','%5.1f')}  {g(qps,'fda2','%7.0f')} | {g(recs,'fda','%5.1f')}  {g(qps,'fda','%7.0f')} | {g(recs,'svf','%5.1f')}  {g(qps,'svf','%7.0f')}")
