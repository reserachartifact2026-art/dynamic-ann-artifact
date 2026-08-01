#!/usr/bin/env python3
import sys, os, re, glob, csv, struct
import numpy as np

ROOT = sys.argv[1]
K = 10
import json

def workload_point_ids(sc, prefer=None):
    """Per-search-step point_id lists from the v11 workload jsonl matching sc.
    SVFusion searches the whole query file; when it has more queries than the GT
    (e.g. t2i 100K vs 10K), results must be indexed by the workload's point_id.
    `prefer` (e.g. '-1m/' or '-10m/') disambiguates when 1m/10m dirs share a
    filename (msmarco-1m/ and msmarco-10m/ both hold msmarco-10m_*.jsonl)."""
    data_root = os.environ["DATA_ROOT"]
    cands = glob.glob(os.path.join(data_root, "freshbang", "workloads", "*", f"{sc}.jsonl"))
    if prefer: cands = sorted(cands, key=lambda p: (prefer not in p, p))
    if not cands: return None
    pids=[]; buf=[]; d=0; ins=False; esc=False
    with open(cands[0]) as f:
        while True:
            c=f.read(1<<20)
            if not c: break
            for ch in c:
                buf.append(ch)
                if esc: esc=False; continue
                if ch=='\\': esc=True; continue
                if ch=='"': ins=not ins; continue
                if ins: continue
                if ch=='{': d+=1
                elif ch=='}':
                    d-=1
                    if d==0:
                        try: o=json.loads(''.join(buf))
                        except: o=None
                        buf=[]
                        if o and o.get("op_type")=="search": pids.append(o.get("point_id"))
    return pids

def fda_cycles(log):
    """fda/fda2 -> list of (ins_thr, del_thr, qps, recall) per search step."""
    if not os.path.exists(log): return None, None
    ins = dele = None; rows = []; rec = []; in_tbl = False
    for ln in open(log, errors="ignore"):
        m = re.search(r"\[PSTEP\] op=(\w+) idx=\d+ n=\d+ ms=[\d.]+ thr=([\d.]+)", ln)
        if m:
            op, thr = m.group(1), float(m.group(2))
            if op == "insert": ins = thr
            elif op == "delete": dele = thr
            elif op == "search": rows.append([ins, dele, thr, None]); ins = dele = None
            continue
        if "Per-batch recall" in ln: in_tbl = True; continue
        if in_tbl:
            mm = re.match(r"\s*(\d+)\s*\|\s*([\d.]+)%", ln)
            if mm: rec.append(float(mm.group(2)))
            elif ln.strip() == "" or ln.startswith("==="): in_tbl = False
    for i in range(min(len(rows), len(rec))): rows[i][3] = rec[i]
    wall = None
    for ln in open(log, errors="ignore"):
        w = re.search(r"Wall time:\s*([\d.]+)\s*ms", ln)
        if w: wall = float(w.group(1))/1000.0
    return rows, wall

def svf_cycles(runlog, binf, gtf, mapf, sc=None, prefer=None):
    if not os.path.exists(runlog): return None, None
    ops = []; scat_n = None
    for ln in open(runlog, errors="ignore"):
        si = re.search(r"Scattered Insert Count = (\d+)", ln)
        if si: scat_n = int(si.group(1)); continue
        m = re.search(r"Operation \d+ \[(INSERT|SEARCH|DELETE)\] Time: ([\d.]+) sec \| Throughput: ([\d.]+)", ln)
        if m:
            op, t, thr = m.group(1).lower(), float(m.group(2)), float(m.group(3))
            if op == "insert" and thr == 0 and scat_n and t > 0:
                thr = scat_n / t
            if op == "insert": scat_n = None
            ops.append((op, t, thr))
    bld = re.search(r"\[BUILD/INSERT\] Time: ([\d.]+)", open(runlog, errors="ignore").read())
    wall = sum(o[1] for o in ops) + (float(bld.group(1)) if bld else 0.0)
    rec = []
    if binf and os.path.exists(binf) and gtf and os.path.exists(gtf):
        gt = np.load(gtf)
        m2b = np.load(mapf) if (mapf and os.path.exists(mapf)) else None
        with open(binf, "rb") as f:
            ns, nq, tk = struct.unpack("III", f.read(12))
            steps = [np.frombuffer(f.read(nq*tk*4), dtype=np.int32).reshape(nq, tk) for _ in range(ns)]
        padded = bool((gt < 0).any())
        pids = workload_point_ids(sc, prefer) if (sc and (nq > gt.shape[1] or padded)) else None
        for s in range(min(ns, gt.shape[0])):
            pid = pids[s] if (pids and s < len(pids) and pids[s]) else list(range(min(nq, gt.shape[1])))
            m = min(len(pid), gt.shape[1]); hit = 0
            for j in range(m):
                pb = steps[s][pid[j], :K]
                if m2b is not None:
                    pb = [int(m2b[x]) for x in pb if 0 <= x < len(m2b)]
                hit += len(set(int(x) for x in pb) & set(gt[s][j, :K].tolist()))
            rec.append(100.0*hit/(m*K) if m else 0.0)
    ins = dele = None; rows = []; si = 0
    for (op, t, thr) in ops:
        if op == "insert": ins = thr
        elif op == "delete": dele = thr
        elif op == "search":
            rows.append([ins, dele, thr, rec[si] if si < len(rec) else None]); ins = dele = None; si += 1
    return rows, wall

comp = open(os.path.join(ROOT, "COMPARISON.csv"), "w", newline="")
cw = csv.writer(comp)
cw.writerow(["dataset","workload","search_step",
             "fda2_ins/s","fda2_del/s","fda2_qps","fda2_recall",
             "fda_ins/s","fda_del/s","fda_qps","fda_recall",
             "svf_ins/s","svf_del/s","svf_qps","svf_recall"])
summ = open(os.path.join(ROOT, "SUMMARY.csv"), "w", newline="")
sw = csv.writer(summ)
sw.writerow(["dataset","workload","search_steps",
             "fda_wall_s","fda_recall","fda_ins/s","fda_del/s","fda_qps",
             "fda2_wall_s","fda2_recall","fda2_ins/s","fda2_del/s","fda2_qps",
             "svf_wall_s","svf_recall","svf_ins/s","svf_del/s","svf_qps"])

def rmean(rows):
    r = [x[3] for x in rows if x and x[3] is not None] if rows else []
    return round(sum(r)/len(r), 2) if r else ""
def tmean(rows, j):
    v = [x[j] for x in rows if x and x[j] is not None] if rows else []
    return round(sum(v)/len(v)) if v else ""

for dsdir in sorted(glob.glob(os.path.join(ROOT, "*"))):
    if not os.path.isdir(dsdir): continue
    ds = os.path.basename(dsdir)
    if ds.startswith("_"): continue
    scen = set()
    for f in glob.glob(os.path.join(dsdir, "fda2_*.log")) + glob.glob(os.path.join(dsdir, "fda_*.log")):
        b = os.path.basename(f); scen.add(re.sub(r"^(fda2?|svf)_", "", b).rsplit(".log",1)[0])
    for f in glob.glob(os.path.join(dsdir, "svf_*.run.log")):
        scen.add(os.path.basename(f)[4:-8])
    for sc in sorted(scen):
        f2, w2 = fda_cycles(os.path.join(dsdir, f"fda2_{sc}.log"))
        f1, w1 = fda_cycles(os.path.join(dsdir, f"fda_{sc}.log"))
        wk = os.path.join(dsdir, "svf_work")
        prefer = "-10m/" if ds.endswith("10m") else "-1m/"
        sv, ws = svf_cycles(os.path.join(dsdir, f"svf_{sc}.run.log"),
                            os.path.join(wk, f"{sc}_results.bin"),
                            os.path.join(wk, f"{sc}_gt.npy"),
                            os.path.join(wk, f"{sc}_int2base.npy"), sc, prefer)
        n = max(len(f2 or []), len(f1 or []), len(sv or []))
        for i in range(n):
            def g(rows, j):
                if rows and i < len(rows) and rows[i][j] is not None: return round(rows[i][j], 2)
                return ""
            cw.writerow([ds, sc, i,
                         g(f2,0), g(f2,1), g(f2,2), g(f2,3),
                         g(f1,0), g(f1,1), g(f1,2), g(f1,3),
                         g(sv,0), g(sv,1), g(sv,2), g(sv,3)])
        W = lambda w: round(w,1) if w else ""
        sw.writerow([ds, sc, n,
                     W(w1), rmean(f1), tmean(f1,0), tmean(f1,1), tmean(f1,2),
                     W(w2), rmean(f2), tmean(f2,0), tmean(f2,1), tmean(f2,2),
                     W(ws), rmean(sv), tmean(sv,0), tmean(sv,1), tmean(sv,2)])
comp.close(); summ.close()
print("wrote", os.path.join(ROOT, "COMPARISON.csv"), "and SUMMARY.csv")
