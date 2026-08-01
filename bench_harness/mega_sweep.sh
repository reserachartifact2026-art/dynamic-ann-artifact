#!/usr/bin/env bash
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
if [ -f "$PROJECT_ROOT/.env" ]; then set -a; source "$PROJECT_ROOT/.env"; set +a; fi
DATA_ROOT="${DATA_ROOT:-/path/to/data}"
RESULTS_ROOT="${RESULTS_ROOT:-${DATA_ROOT}/results}"
export PROJECT_ROOT DATA_ROOT RESULTS_ROOT
MEGA="$RESULTS_ROOT/MEGA_SWEEP"
BENCH="${PROJECT_ROOT}/bench_harness"
GB="${DATA_ROOT}/freshbang_data"
FB="${DATA_ROOT}/freshbang"
FDADIR="${PROJECT_ROOT}/FreshDiskANN-Naive"
FDABIN="${PROJECT_ROOT}/FreshDiskANN-Naive/bin/fda_fresh_diskann"
FDA2DIR="${PROJECT_ROOT}/FreshDiskANN-Naive/FreshDiskANN_sequentialModel"
FDA2BIN="$FDA2DIR/bin/fda2_fresh_diskann"
G=${GPU:-0}
mkdir -p "$MEGA"; cd "$BENCH"
echo "MEGA_SWEEP start $(date)  GPU=$G  -> $MEGA"

for d in sift deep glove msturing msmarco t2i wiki  deep10m msturing10m msmarco10m t2i10m; do
  echo "===== SWEEP $d ($(date)) ====="
  RESULTS="$MEGA/$d" GPU_FDA=$G GPU_FDA2=$G GPU_SVF=$G ./bench.sh -d "$d" -s fda,fda2,svf -a all -L 100 \
      > "$MEGA/_log_$d.txt" 2>&1 || echo "  $d exited $?"
done

echo "===== 100M msturing-100m ($(date)) ====="
RESULTS="$MEGA/msturing100m" GPU_SVF=$G ./bench.sh -d msturing100m -s svf -a all -L 100 > "$MEGA/_log_msturing100m_svf.txt" 2>&1 || true
( cd "$FDADIR" && sed -i -E "s/^#define D .*/#define D 100/" baseline_src/vamana.h && make compile >/dev/null 2>&1 )
mkdir -p "$MEGA/msturing100m"
for w in expiration_time interleaved msturing_ih sliding_window; do
  echo "  fda msturing100m $w"
  CUDA_VISIBLE_DEVICES=$G FDA_SKIP_MERGE=1 "$FDABIN" $GB/vamana_msturing_50m.out $FB/workloads/msturing-100m/msturing-100m_$w.jsonl $FB/data/msturing_100m/query.fbin $GB/msturing_gt_placeholder.bin --searchL 100 --k 10 --base-data $FB/data/msturing_100m/base.fbin --metric l2 --m 51 --t 1000000 --insert-batch 2000 --query-batch 2000 > "$MEGA/msturing100m/fda_msturing-100m_$w.log" 2>&1 || echo "    exit $?"
done
( cd "$FDA2DIR" && sed -i -E "s/^#define D .*/#define D 100/" baseline_src/vamana.h && make compile P=8 >/dev/null 2>&1 )
for w in expiration_time interleaved msturing_ih sliding_window; do
  echo "  fda2 msturing100m $w"
  CUDA_VISIBLE_DEVICES=$G FDA2_MEDOID_LAST=1 "$FDA2BIN" $GB/vamana_msturing_50m.out $FB/workloads/msturing-100m/msturing-100m_$w.jsonl $FB/data/msturing_100m/query.fbin $GB/msturing_gt_placeholder.bin --searchL 100 --k 10 --base-data $FB/data/msturing_100m/base.fbin --metric l2 --insert-batch 10000 --query-batch 10000 --delete-batch 10000 > "$MEGA/msturing100m/fda2_msturing-100m_$w.log" 2>&1 || echo "    exit $?"
done

echo "===== 100M deep-100m ($(date)) ====="
RESULTS="$MEGA/deep100m" GPU_SVF=$G ./bench.sh -d deep100m -s svf -w sliding_window -a all -L 100 > "$MEGA/_log_deep100m_svf.txt" 2>&1 || true
( cd "$FDADIR" && sed -i -E "s/^#define D .*/#define D 96/" baseline_src/vamana.h && make compile >/dev/null 2>&1 )
mkdir -p "$MEGA/deep100m"
CUDA_VISIBLE_DEVICES=$G FDA_SKIP_MERGE=1 "$FDABIN" $GB/vamana_deep_50m.out $FB/workloads/deep-100m/deep-100m_sliding_window.jsonl $FB/data/deep_10m/query.fbin $GB/deep_gt_placeholder.bin --searchL 100 --k 10 --base-data $FB/data/deep_100m/base.fbin --metric l2 --m 51 --t 1000000 --insert-batch 2000 --query-batch 2000 > "$MEGA/deep100m/fda_deep-100m_sliding_window.log" 2>&1 || echo "    fda exit $?"
( cd "$FDA2DIR" && sed -i -E "s/^#define D .*/#define D 96/" baseline_src/vamana.h && make compile P=8 >/dev/null 2>&1 )
CUDA_VISIBLE_DEVICES=$G FDA2_MEDOID_LAST=1 "$FDA2BIN" $GB/vamana_deep_50m.out $FB/workloads/deep-100m/deep-100m_sliding_window.jsonl $FB/data/deep_10m/query.fbin $GB/deep_gt_placeholder.bin --searchL 100 --k 10 --base-data $FB/data/deep_100m/base.fbin --metric l2 --insert-batch 10000 --query-batch 10000 --delete-batch 10000 > "$MEGA/deep100m/fda2_deep-100m_sliding_window.log" 2>&1 || echo "    fda2 exit $?"

echo "===== CONSOLIDATE ($(date)) ====="
for d in "$MEGA"/*/; do python3 consolidate.py "$d" >/dev/null 2>&1 || true; done
python3 "$BENCH/mega_report.py" "$MEGA" > "$MEGA/MASTER_SUMMARY.csv" 2>"$MEGA/_report_err.txt" || true
echo "MEGA_SWEEP_DONE $(date)"
