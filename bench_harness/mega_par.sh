#!/usr/bin/env bash
set -uo pipefail
GPU="$1"; export PROOT="$2"; DS_LIST="$3"; DS100="${4:-}"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
if [ -f "$PROJECT_ROOT/.env" ]; then set -a; source "$PROJECT_ROOT/.env"; set +a; fi
DATA_ROOT="${DATA_ROOT:-/path/to/data}"
RESULTS_ROOT="${RESULTS_ROOT:-${DATA_ROOT}/results}"
export PROJECT_ROOT DATA_ROOT RESULTS_ROOT
BIG="${DATA_ROOT}"
MEGA="$RESULTS_ROOT/MEGA_SWEEP"
BENCH="${PROJECT_ROOT}/bench_harness"
FB="${DATA_ROOT}/freshbang"
GB="${DATA_ROOT}/freshbang_data"
export SKIP_SVF_COMPILE=1
FDADIR="${PROJECT_ROOT}/FreshDiskANN-Naive"
FDABIN="${PROJECT_ROOT}/FreshDiskANN-Naive/bin/fda_fresh_diskann"
FDA2DIR="${PROJECT_ROOT}/FreshDiskANN-Naive/FreshDiskANN_sequentialModel"
FDA2BIN="$FDA2DIR/bin/fda2_fresh_diskann"
mkdir -p "$MEGA"; cd "$BENCH"
echo "[GPU$GPU] START $(date) PROOT=$PROOT ds=$DS_LIST ds100=$DS100"

for d in ${DS_LIST//,/ }; do
  echo "[GPU$GPU] ===== $d $(date) ====="
  RESULTS="$MEGA/$d" GPU_FDA=$GPU GPU_FDA2=$GPU GPU_SVF=$GPU \
    ./bench.sh -d "$d" -s fda,fda2,svf -a all -L 100 > "$MEGA/_log_${d}_gpu${GPU}.txt" 2>&1 || echo "  $d exit $?"
  python3 consolidate.py "$MEGA/$d" >/dev/null 2>&1 || true
done

for k in ${DS100//,/ }; do
  echo "[GPU$GPU] ===== 100M $k $(date) ====="
  case "$k" in
    msturing100m) D=100; seed=$GB/vamana_msturing_50m.out; base=$FB/data/msturing_100m/base.fbin; query=$FB/data/msturing_100m/query.fbin; gt=$GB/msturing_gt_placeholder.bin; dir=msturing-100m; wls="expiration_time interleaved msturing_ih sliding_window";;
    deep100m)     D=96;  seed=$GB/vamana_deep_50m.out;     base=$FB/data/deep_100m/base.fbin;     query=$FB/data/deep_10m/query.fbin;     gt=$GB/deep_gt_placeholder.bin;     dir=deep-100m;     wls="sliding_window";;
    *) echo "  unknown 100m key $k"; continue;;
  esac
  R="$MEGA/$k"; mkdir -p "$R"
  RESULTS="$R" GPU_SVF=$GPU ./bench.sh -d "$k" -s svf -a all -L 100 > "$MEGA/_log_${k}_svf_gpu${GPU}.txt" 2>&1 || echo "  svf exit $?"
  ( cd "$FDADIR" && sed -i -E "s/^#define D .*/#define D $D/" baseline_src/vamana.h && make compile >/dev/null 2>&1 )
  for w in $wls; do
    echo "  [GPU$GPU] fda $dir $w"
    CUDA_VISIBLE_DEVICES=$GPU FDA_SKIP_MERGE=1 "$FDABIN" "$seed" "$FB/workloads/$dir/${dir}_$w.jsonl" "$query" "$gt" \
      --searchL 100 --k 10 --base-data "$base" --metric l2 --m 51 --t 1000000 \
      --insert-batch 2000 --query-batch 2000 > "$R/fda_${dir}_$w.log" 2>&1 || echo "    fda $w exit $?"
  done
  ( cd "$FDA2DIR" && sed -i -E "s/^#define D .*/#define D $D/" baseline_src/vamana.h && make compile P=8 >/dev/null 2>&1 )
  for w in $wls; do
    echo "  [GPU$GPU] fda2 $dir $w"
    CUDA_VISIBLE_DEVICES=$GPU FDA2_MEDOID_LAST=1 "$FDA2BIN" "$seed" "$FB/workloads/$dir/${dir}_$w.jsonl" "$query" "$gt" \
      --searchL 100 --k 10 --base-data "$base" --metric l2 \
      --insert-batch 10000 --query-batch 10000 --delete-batch 10000 > "$R/fda2_${dir}_$w.log" 2>&1 || echo "    fda2 $w exit $?"
  done
  python3 consolidate.py "$R" >/dev/null 2>&1 || true
done
echo "[GPU$GPU] DONE $(date)"
