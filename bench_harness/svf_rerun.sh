#!/usr/bin/env bash
set -o pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
if [ -f "$PROJECT_ROOT/.env" ]; then set -a; source "$PROJECT_ROOT/.env"; set +a; fi
DATA_ROOT="${DATA_ROOT:-/path/to/data}"
RESULTS_ROOT="${RESULTS_ROOT:-${DATA_ROOT}/results}"
export PROJECT_ROOT DATA_ROOT RESULTS_ROOT
BENCH="${PROJECT_ROOT}/bench_harness"
MEGA="$RESULTS_ROOT/MEGA_SWEEP"
cd "$BENCH"

unset SKIP_SVF_COMPILE
./bench.sh -d deep -s svf -a compile -L 100 > /tmp/svf_rerun_build.log 2>&1
if grep -q "Built target benchmark" /tmp/svf_rerun_build.log; then echo SVF_REBUILT_OK; else echo SVF_REBUILD_FAIL; fi

export SKIP_SVF_COMPILE=1
( RESULTS="$MEGA/deep"    GPU_SVF=0 ./bench.sh -d deep    -s svf -a all -L 100 > /tmp/rerun_deep.log 2>&1
  RESULTS="$MEGA/deep10m" GPU_SVF=0 ./bench.sh -d deep10m -s svf -a all -L 100 > /tmp/rerun_deep10m.log 2>&1
  echo GPU0_RERUN_DONE ) &
( RESULTS="$MEGA/t2i"    GPU_SVF=1 ./bench.sh -d t2i    -s svf -a all -L 100 > /tmp/rerun_t2i.log 2>&1
  RESULTS="$MEGA/t2i10m" GPU_SVF=1 ./bench.sh -d t2i10m -s svf -a all -L 100 > /tmp/rerun_t2i10m.log 2>&1
  echo GPU1_RERUN_DONE ) &
wait
for d in deep deep10m t2i t2i10m; do python3 consolidate.py "$MEGA/$d" >/dev/null 2>&1 || true; done
echo SVF_RERUN_ALL_DONE
