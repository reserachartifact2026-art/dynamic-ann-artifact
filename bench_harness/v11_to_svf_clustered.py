#!/usr/bin/env python3
import json, sys, os, numpy as np
wl, ds_name, runbook_out, gt_out, k, idsdir, map_out = (
    sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6], sys.argv[7])
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

events=list(objs(wl))
ins_ids=[e["vector_ids"] for e in events if e.get("op_type")=="insert"]
base_off=min(min(v) for v in ins_ids)
max_base=max(max(v) for v in ins_ids)+1
seed = base_off if base_off>0 else int(os.environ.get("SVF_CLUSTERED_SEED", str(max_base//2)))

base2int={}
for b in range(seed): base2int[b]=b
internal=seed
int2base=list(range(seed))

ops=[]; gts=[]; nbatch=0
for e in events:
    op=e.get("op_type")
    if op=="insert":
        v=[b for b in e["vector_ids"] if b>=seed]
        if not v: continue
        nbatch+=1
        fn=os.path.join(idsdir, f"ins_{nbatch:04d}.i64")
        np.asarray(v, dtype=np.int64).tofile(fn)
        for b in v:
            base2int[b]=internal; int2base.append(b); internal+=1
        ops.append(("insert_ids", fn))
    elif op=="delete":
        v=e["vector_ids"]
        iids=[base2int[b] for b in v if b in base2int]
        fn=os.path.join(idsdir, f"del_{nbatch:04d}_{len(ops)}.i64")
        np.asarray(sorted(iids), dtype=np.int64).tofile(fn)
        ops.append(("delete_ids", fn))
    elif op=="search":
        ops.append(("search", None))
        gts.append(np.asarray([row[:k] for row in e["groundtruth"]], dtype=np.int32))

np.save(map_out, np.asarray(int2base, dtype=np.int64))
lines=[]; step=1
lines.append(f"  {step}:\n    operation: insert\n    start: 0\n    end: {seed}"); step+=1
for (op,fn) in ops:
    if op=="search":
        lines.append(f"  {step}:\n    operation: search\n    qps: 30\n    k: {k}")
    elif op=="insert_ids":
        lines.append(f"  {step}:\n    operation: insert\n    ids_file: {fn}")
    elif op=="delete_ids":
        lines.append(f"  {step}:\n    operation: delete\n    ids_file: {fn}")
    step+=1
if gts:
    gt_arr=np.stack(gts,0); np.save(gt_out, gt_arr)
else:
    gt_arr=np.zeros((0,)); np.save(gt_out, gt_arr)
with open(runbook_out,"w") as f:
    results_root = os.environ["RESULTS_ROOT"]
    f.write(f"{ds_name}:\n  max_pts: {max(internal,max_base,1000000)}\n  res_path: {results_root}/svfusion_v11/\n  qps: 250.0\n  batch_size: 4\n")
    f.write("\n".join(lines)+"\n")
print(f"base_off={base_off} seed={seed} total_internal={internal} ops={step-1} searches={len(gts)} GT={gt_arr.shape}")
print(f"insert_ids batches={sum(1 for o in ops if o[0]=='insert_ids')} delete_ids batches={sum(1 for o in ops if o[0]=='delete_ids')}")
print(f"int2base saved: {map_out} (len={len(int2base)})")
