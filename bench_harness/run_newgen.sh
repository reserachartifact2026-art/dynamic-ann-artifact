#!/usr/bin/env bash
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
if [ -f "$PROJECT_ROOT/.env" ]; then set -a; source "$PROJECT_ROOT/.env"; set +a; fi
DATA_ROOT="${DATA_ROOT:-/path/to/data}"
RESULTS_ROOT="${RESULTS_ROOT:-${DATA_ROOT}/results}"
export PROJECT_ROOT DATA_ROOT RESULTS_ROOT
BENCH="${PROJECT_ROOT}/bench_harness"
GB="${DATA_ROOT}/freshbang_data"
FB="${DATA_ROOT}/freshbang"
FDA2DIR="${PROJECT_ROOT}/FreshDiskANN-Naive/FreshDiskANN_sequentialModel"
FDABIN="${PROJECT_ROOT}/FreshDiskANN-Naive/bin/fda_fresh_diskann"
FDA2BIN="$FDA2DIR/bin/fda2_fresh_diskann"
R="$RESULTS_ROOT/newgen"
GTM=$GB/msturing_gt_placeholder.bin; GTD=$GB/deep_gt_placeholder.bin
mkdir -p $R
cd $BENCH

echo "===== 1/4 text2image-10m all 5, all 3 (IP D=201) ====="
RESULTS=$R/t2i10m GPU_FDA=0 GPU_FDA2=0 GPU_SVF=0 ./bench.sh -d t2i10m -s fda,fda2,svf -a all -L 100

echo "===== 2/4 msturing-1m clustered fda (--m 2, D=100) ====="
sed -i -E 's/^#define D .*/#define D 100/' "$(dirname "$FDABIN")/../baseline_src/vamana.h"
( cd "$(dirname "$FDABIN")/.." && make compile >/tmp/newgen_fda_d100.log 2>&1 )
mkdir -p $R/msturing_cl
CUDA_VISIBLE_DEVICES=0 "$FDABIN" $GB/vamana_msturing_500k.out $FB/workloads/msturing-1m/msturing-100m_clustered.jsonl $FB/data/msturing_100m/query.fbin $GTM --searchL 100 --k 10 --base-data $GB/msturing_1m_base.fbin --metric l2 --m 2 > $R/msturing_cl/fda_msturing-100m_clustered.log 2>&1 || echo "  msturing clustered fda exit=$?"

echo "===== 3/4 msturing-100m fda2 (P_EXTRA=16, batch 2000, D=100) ====="
( cd "$FDA2DIR" && sed -i -E 's/^#define D .*/#define D 100/' baseline_src/vamana.h && make compile P=16 >/tmp/newgen_fda2_m100.log 2>&1 )
mkdir -p $R/msturing100m
for w in expiration_time interleaved msturing_ih sliding_window; do
  echo "  msturing-100m fda2 $w"
  CUDA_VISIBLE_DEVICES=0 FDA2_MEDOID_LAST=1 "$FDA2BIN" $GB/vamana_msturing_50m.out $FB/workloads/msturing-100m/msturing-100m_$w.jsonl $FB/data/msturing_100m/query.fbin $GTM --searchL 100 --k 10 --base-data $FB/data/msturing_100m/base.fbin --metric l2 --insert-batch 2000 --query-batch 2000 --delete-batch 2000 > $R/msturing100m/fda2_msturing-100m_$w.log 2>&1 || echo "    exit=$?"
done

echo "===== 4/4 deep-100m fda2 (P_EXTRA=16, batch 2000, D=96) ====="
while [ ! -s $GB/vamana_deep_50m.out ]; do sleep 30; done
( cd "$FDA2DIR" && sed -i -E 's/^#define D .*/#define D 96/' baseline_src/vamana.h && make compile P=16 >/tmp/newgen_fda2_d96.log 2>&1 )
mkdir -p $R/deep100m
CUDA_VISIBLE_DEVICES=0 FDA2_MEDOID_LAST=1 "$FDA2BIN" $GB/vamana_deep_50m.out $FB/workloads/deep-100m/deep-100m_sliding_window.jsonl $FB/data/deep_10m/query.fbin $GTD --searchL 100 --k 10 --base-data $FB/data/deep_100m/base.fbin --metric l2 --insert-batch 2000 --query-batch 2000 --delete-batch 2000 > $R/deep100m/fda2_deep-100m_sliding_window.log 2>&1 || echo "  deep-100m fda2 exit=$?"

echo "===== NEWGEN_ALL_DONE ====="
