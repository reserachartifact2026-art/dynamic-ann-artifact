#!/usr/bin/env bash
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
if [ -f "$PROJECT_ROOT/.env" ]; then set -a; source "$PROJECT_ROOT/.env"; set +a; fi
DATA_ROOT="${DATA_ROOT:-/path/to/data}"
RESULTS_ROOT="${RESULTS_ROOT:-${DATA_ROOT}/results}"
export PROJECT_ROOT DATA_ROOT RESULTS_ROOT
BIG="${DATA_ROOT}"
FB="${DATA_ROOT}/freshbang"
BENCH="${PROJECT_ROOT}/bench_harness"; PFX=$BENCH/prefix; mkdir -p "$PFX"
DA="${DISKANN_BIN:-${PROJECT_ROOT}/DiskANN/build/apps/build_memory_index}"
ST="$RESULTS_ROOT/msm5m_diskann_status.txt"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$ST"; }
OUT=$BIG/freshbang_data/vamana_msmarco_5m_ip769.out
[ -f "$OUT" ] && { log "graph already exists: $OUT"; exit 0; }

RAW=$PFX/msmarco_5m_base.fbin
AUG=$PFX/msmarco_5m_aug769.fbin
DG=$BIG/freshbang_data/diskann_msm_mips_5m

log "start (CPU msmarco 5M DiskANN-mips seed)"
[ -f "$RAW" ] || { log "5M raw prefix (768-d)"; python3 $BENCH/make_fbin_prefix.py "$FB/data/msmarco_10m/base.fbin" "$RAW" 5000000 >>"$ST" 2>&1; }
[ -f "$AUG" ] || { log "augment 5M (M over full base)";  python3 $BENCH/aug_msmarco_5m.py >>"$ST" 2>&1; }
log "DiskANN mips build (5M, -T 16, CPU)"
"$DA" --data_type float --dist_fn mips --data_path "$RAW" \
  --index_path_prefix "$DG" -R 64 -L 100 -T 16 --alpha 1.2 >>"$ST" 2>&1
log "  DiskANN build exit=$?"
log "convert -> $OUT"
python3 $BENCH/diskann_to_fda_out.py "$AUG" "$DG" "$OUT" 64 >>"$ST" 2>&1
log "MSM5M_DISKANN_DONE exit=$? -> $OUT"
