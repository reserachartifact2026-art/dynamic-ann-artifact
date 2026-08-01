#!/usr/bin/env python3
import sys, os, re, glob
MEGA = sys.argv[1]

def num(x):
    try: return float(x)
    except: return ''

def parse_fda(path):
    """fda / fda2 log -> dict of metrics."""
    d = dict(recall='', ins_s='', qps='', del_s='', insert_ms='', search_ms='', delete_ms='',
             merges='', merge_ms='', total_wall_s='')
    ins=[]; qps=[]; dele=[]; wall_direct=''
    for ln in open(path, errors='ignore'):
        m = re.search(r'10-recall@10:\s*([\d.]+)', ln)
        if m: d['recall'] = m.group(1)
        m = re.search(r'Insert(?: phase)?:\s*([\d.]+) ms \(([\d.]+) ins/sec\)', ln)
        if m: d['insert_ms'], d['ins_s'] = m.group(1), m.group(2)
        m = re.search(r'Search(?: phase)?:\s*([\d.]+) ms \(([\d.]+) q/sec\)', ln)
        if m: d['search_ms'], d['qps'] = m.group(1), m.group(2)
        m = re.search(r'Delete(?: phase)?:\s*([\d.]+) ms \(([\d.]+) (?:d|del)/sec\)', ln)
        if m: d['delete_ms'], d['del_s'] = m.group(1), m.group(2)
        m = re.search(r'Wall time:\s*([\d.]+) ms', ln)
        if m: wall_direct = m.group(1)
        m = re.search(r'Merges fired:\s*(\d+)', ln)
        if m: d['merges'] = m.group(1)
        m = re.search(r'Merge phase:\s*([\d.]+) ms', ln)
        if m: d['merge_ms'] = m.group(1)
        p = re.search(r'\[PSTEP\] op=(\w+).*thr=([\d.]+)', ln)
        if p:
            (ins if p.group(1)=='insert' else dele if p.group(1)=='delete' else qps if p.group(1)=='search' else []).append(float(p.group(2)))
    mn=lambda a:round(sum(a)/len(a)) if a else ''
    if d['ins_s']=='' : d['ins_s']=mn(ins)
    if d['qps']==''   : d['qps']=mn(qps)
    if d['del_s']=='' : d['del_s']=mn(dele)
    if wall_direct:
        d['total_wall_s'] = round(float(wall_direct)/1000.0, 1)
    else:
        tw = sum(float(d[k]) for k in ('insert_ms','search_ms','delete_ms','merge_ms') if d[k])
        d['total_wall_s'] = round(tw/1000.0, 1) if tw else ''
    return d

def parse_svf(path):
    """svf score.txt -> dict."""
    d = dict(recall='', ins_s='', qps='', del_s='', insert_ms='', search_ms='', delete_ms='',
             merges='', merge_ms='', total_wall_s='')
    if not os.path.exists(path): return None
    txt = open(path, errors='ignore').read()
    m = re.search(r'mean=([\d.]+)%', txt)
    if m: d['recall'] = m.group(1)
    m = re.search(r'final:\s*([\d.]+)%', txt)
    if m: d['recall_final'] = m.group(1)
    m = re.search(r'INSERT batches=\d+ mean=([\d.]+)', txt)
    if m: d['ins_s'] = m.group(1)
    m = re.search(r'DELETE batches=\d+ mean=([\d.]+)', txt)
    if m: d['del_s'] = m.group(1)
    m = re.search(r'SEARCH steps=\d+ mean=([\d.]+)', txt)
    if m: d['qps'] = m.group(1)
    m = re.search(r'insert=([\d.]+)s delete=([\d.]+)s search=([\d.]+)s', txt)
    if m:
        d['insert_ms'], d['delete_ms'], d['search_ms'] = str(float(m.group(1))*1000), str(float(m.group(2))*1000), str(float(m.group(3))*1000)
        d['total_wall_s'] = round(float(m.group(1))+float(m.group(2))+float(m.group(3)), 1)
    return d

cols = ['folder','dataset_workload','system','recall','ins_s','qps','del_s',
        'insert_ms','search_ms','delete_ms','merges','merge_ms','total_wall_s']
print(','.join(cols))
for folder in sorted(os.listdir(MEGA)):
    fp = os.path.join(MEGA, folder)
    if not os.path.isdir(fp): continue
    for sysname in ('fda','fda2'):
        for log in sorted(glob.glob(os.path.join(fp, f'{sysname}_*.log'))):
            scen = os.path.basename(log)[len(sysname)+1:-4]
            d = parse_fda(log)
            print(','.join(str(x) for x in [folder, scen, sysname, d['recall'], d['ins_s'], d['qps'], d['del_s'],
                  d['insert_ms'], d['search_ms'], d['delete_ms'], d['merges'], d['merge_ms'], d['total_wall_s']]))
    for sc in sorted(glob.glob(os.path.join(fp, 'svf_*.score.txt'))):
        scen = os.path.basename(sc)[4:-10]
        d = parse_svf(sc)
        if d is None: continue
        print(','.join(str(x) for x in [folder, scen, 'svf', d['recall'], d['ins_s'], d['qps'], d['del_s'],
              d['insert_ms'], d['search_ms'], d['delete_ms'], d['merges'], d['merge_ms'], d['total_wall_s']]))
