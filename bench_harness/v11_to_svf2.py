#!/usr/bin/env python3
import json, sys, os, numpy as np

wl, ds_name, runbook_out, gt_out, k, idsdir = (
    sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6])
os.makedirs(idsdir, exist_ok=True)

def objs(path):
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
                        s=''.join(buf).strip(); buf=[]
                        if s: yield json.loads(s)

def is_contig(v):
    return v == list(range(v[0], v[0]+len(v)))

ops=[]; gts=[]; first_ins=None; del_batch=0
max_id=0
for e in objs(wl):
    op=e.get("op_type")
    if op=="insert":
        v=e["vector_ids"]
        assert is_contig(v), "insert must be contiguous"
        if first_ins is None: first_ins=v[0]
        ops.append(("insert", v[0], v[-1]+1, None))
        max_id=max(max_id, v[-1]+1)
    elif op=="delete":
        v=e["vector_ids"]
        if is_contig(v):
            ops.append(("delete", v[0], v[-1]+1, None))
        else:
            del_batch+=1
            fn=os.path.join(idsdir, f"del_{del_batch:04d}.i64")
            np.asarray(sorted(v), dtype=np.int64).tofile(fn)
            ops.append(("delete_ids", None, None, fn))
        max_id=max(max_id, max(v)+1)
    elif op=="search":
        ops.append(("search", None, None, None))
        gts.append(np.asarray([row[:k] for row in e["groundtruth"]], dtype=np.int32))

base_off=first_ins
lines=[]; step=1
lines.append(f"  {step}:\n    operation: insert\n    start: 0\n    end: {base_off}"); step+=1
for (op,s,en,fn) in ops:
    if op=="search":
        lines.append(f"  {step}:\n    operation: search\n    qps: 30\n    k: {k}")
    elif op=="insert":
        lines.append(f"  {step}:\n    operation: insert\n    start: {s}\n    end: {en}")
    elif op=="delete":
        lines.append(f"  {step}:\n    operation: delete\n    start: {s}\n    end: {en}")
    elif op=="delete_ids":
        lines.append(f"  {step}:\n    operation: delete\n    ids_file: {fn}")
    step+=1

maxnq=max(g.shape[0] for g in gts)
gt_arr=np.full((len(gts),maxnq,gts[0].shape[1]),-1,dtype=np.int32)
for _i,_g in enumerate(gts): gt_arr[_i,:_g.shape[0],:]=_g
np.save(gt_out, gt_arr)
with open(runbook_out,"w") as f:
    results_root = os.environ["RESULTS_ROOT"]
    f.write(f"{ds_name}:\n  max_pts: {max(max_id,1000000)}\n  res_path: {results_root}/svfusion_v11/\n  qps: 250.0\n  batch_size: 4\n")
    f.write("\n".join(lines)+"\n")

n_range_del=sum(1 for o in ops if o[0]=="delete")
n_scat_del =sum(1 for o in ops if o[0]=="delete_ids")
print(f"base_offset={base_off}  ops={step-1} (incl BUILD)  searches={gt_arr.shape[0]}  GT={gt_arr.shape}")
print(f"deletes: range={n_range_del} scattered(ids_file)={n_scat_del}  idsdir={idsdir}")
