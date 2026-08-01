#!/usr/bin/env bash
set -uo pipefail

SELFDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
if [ -f "$PROJECT_ROOT/.env" ]; then set -a; source "$PROJECT_ROOT/.env"; set +a; fi
DATA_ROOT="${DATA_ROOT:-/path/to/data}"
RESULTS_ROOT="${RESULTS_ROOT:-${DATA_ROOT}/results}"
export PROJECT_ROOT DATA_ROOT RESULTS_ROOT
BIG="${DATA_ROOT}"
FB="${DATA_ROOT}/freshbang"
PROOT="$PROJECT_ROOT"
ROOT_FDA="${PROJECT_ROOT}/FreshDiskANN-Naive"
ROOT_FDA2="$ROOT_FDA/FreshDiskANN_sequentialModel"
BUILDER="${PROJECT_ROOT}/BANG-Variants-vamana-gpu"
SVF="${PROJECT_ROOT}/svfusion/examples"
BIN_FDA="${PROJECT_ROOT}/FreshDiskANN-Naive/bin/fda_fresh_diskann"
BIN_FDA2="$ROOT_FDA2/bin/fda2_fresh_diskann"
BIN_SVF="$SVF/build/benchmark"

DATASET=""; SYSTEMS="all"; ACTION="all"; WL="all"; L=100; K=10
while getopts "d:s:a:w:L:k:h" opt; do case $opt in
  d) DATASET=$OPTARG;; s) SYSTEMS=$OPTARG;; a) ACTION=$OPTARG;;
  w) WL=$OPTARG;; L) L=$OPTARG;; k) K=$OPTARG;;
  h) printf '%s\n' 'Usage: bench.sh -d <dataset> [-s fda,fda2,svf|all] [-a compile|graph|run|all] [-w <workload>|all] [-L <searchL>] [-k <topk>]'; exit 0;; *) exit 1;; esac; done
[ -z "$DATASET" ] && { echo "ERROR: -d <dataset> required (see -h)"; exit 1; }

GPU_FDA=${GPU_FDA:-0}; GPU_FDA2=${GPU_FDA2:-1}; GPU_SVF=${GPU_SVF:-0}
GPU_FRESHBANG=${GPU_FRESHBANG:-0}
BIN_FRESHBANG="${PROJECT_ROOT}/freshbang/v11_drivers/freshbang_v11_driver"
FB_LDPATH="/tmp/freshbang_private:/tmp/freshbang:/mnt/ssd_volume/mumba1/envs/cuvs_bench_26_02/lib"

resolve() {
  EXTRA_FDA=""; PREFIX=0; PREFIX_N=1000000; SVF_DTYPE=float
  case "$1" in
    sift)
      D=128; METRIC=l2; SVF_DS=sift-fb; BASE_OFFSET=500000
      GRAPH="$BIG/Dynamic_Workloads/graph_outputs/vamana_500k.out"
      BASE="$BIG/freshbang_data/sift_1m_base.fbin"
      QUERY="$BIG/Dynamic_Workloads/data/sift1m_query.bin"
      GT="$BIG/Dynamic_Workloads/data/sift1m_groundtruth.bin"
      WLDIR="$FB/workloads/sift-1m";;
    deep)
      D=96; METRIC=l2; SVF_DS=deep-fb; BASE_OFFSET=500000
      GRAPH="$BIG/freshbang_data/vamana_deep_500k.out"
      BASE="$FB/data/deep_10m/base.fbin"; QUERY="$FB/data/deep_10m/query.fbin"
      GT="$BIG/freshbang_data/deep_gt_placeholder.bin"
      WLDIR="$FB/workloads/deep-1m";;
    deep10m)
      D=96; METRIC=l2; SVF_DS=deep-fb; BASE_OFFSET=5000000
      GRAPH="$BIG/freshbang_data/vamana_deep_5m.out"
      BASE="$FB/data/deep_10m/base.fbin"; QUERY="$FB/data/deep_10m/query.fbin"
      GT="$BIG/freshbang_data/deep_gt_placeholder.bin"
      WLDIR="$FB/workloads/deep-10m";;
    glove)
      D=100; METRIC=cosine; SVF_DS=glove-fb; BASE_OFFSET=600000
      GRAPH="$BIG/freshbang_data/vamana_glove_600k_norm.out"
      BASE="$BIG/freshbang_data/glove_base_norm.fbin"
      QUERY="$BIG/freshbang_data/glove_query_norm.fbin"
      GT="$BIG/freshbang_data/glove_gt_placeholder.bin"
      WLDIR="$FB/workloads/glove-100-1m"; EXTRA_FDA="--m 2";;
    t2i)
      D=201; METRIC=ip; SVF_DS=t2i-fb; BASE_OFFSET=500000
      GRAPH="$BIG/freshbang_data/vamana_t2i_500k_aug201.out"
      BASE="$BIG/freshbang_data/t2i_1m_raw.fbin"
      QUERY="$FB/data/text2image_10m/query.fbin"
      GT="$BIG/freshbang_data/t2i_gt_placeholder.bin"
      WLDIR="$FB/workloads/text2image-1m";;
    t2i10m)
      D=201; METRIC=ip; SVF_DS=t2i-fb; BASE_OFFSET=5000000
      GRAPH="$BIG/freshbang_data/vamana_t2i_5m_aug201.out"
      BASE="$FB/data/text2image_10m/base.fbin"
      QUERY="$FB/data/text2image_10m/query.fbin"
      GT="$BIG/freshbang_data/t2i_gt_placeholder.bin"
      WLDIR="$FB/workloads/text2image-10m";;
    wiki)
      D=769; METRIC=ip; SVF_DS=wiki-fb; BASE_OFFSET=500000
      GRAPH="$BIG/freshbang_data/vamana_wiki_500k_aug769.out"
      BASE="$BIG/freshbang_data/wiki_1m_base.fbin"; QUERY="$FB/data/wikipedia_10m/query.fbin"
      GT="$BIG/freshbang_data/wiki_gt_placeholder.bin"
      WLDIR="$FB/workloads/wikipedia-1m";;
    wiki10m)
      D=769; METRIC=ip; SVF_DS=wiki-fb; BASE_OFFSET=5000000
      GRAPH="$BIG/freshbang_data/vamana_wiki_5m_aug769.out"
      BASE="$FB/data/wikipedia_10m/base.fbin"; QUERY="$FB/data/wikipedia_10m/query.fbin"
      GT="$BIG/freshbang_data/wiki_gt_placeholder.bin"
      WLDIR="$FB/workloads/wikipedia-10m";;
    msturing)
      D=100; METRIC=l2; SVF_DS=msturing-fb; BASE_OFFSET=500000; PREFIX=1
      GRAPH="$BIG/freshbang_data/vamana_msturing_500k.out"
      BASE="$FB/data/msturing_100m/base.fbin"; QUERY="$FB/data/msturing_100m/query.fbin"
      GT="$BIG/freshbang_data/msturing_gt_placeholder.bin"
      WLDIR="$FB/workloads/msturing-1m";;
    msmarco)
      D=769; METRIC=ip; SVF_DS=msmarco-fb; BASE_OFFSET=500000
      GRAPH="$BIG/freshbang_data/msmarco_1m_seed.out"
      BASE="$BIG/freshbang_data/msmarco_1m_base.fbin"; QUERY="$FB/data/msmarco_10m/query.fbin"
      GT="$BIG/freshbang_data/msmarco_gt_placeholder.bin"
      WLDIR="$FB/workloads/msmarco-1m";;
    msmarcol2)
      D=768; METRIC=l2; SVF_DS=msmarco-fb-norm; BASE_OFFSET=500000
      GRAPH="$BIG/freshbang_data/msmarco_1m_seed_l2norm.out"
      BASE="$BIG/freshbang_data/msmarco_1m_base_l2norm.fbin"
      QUERY="$BIG/freshbang_data/msmarco_1m_query_l2norm.fbin"
      GT="$BIG/freshbang_data/msmarco_gt_placeholder.bin"
      WLDIR="$FB/workloads/msmarco-1m";;
    msturing10m)
      D=100; METRIC=l2; SVF_DS=msturing-fb; BASE_OFFSET=5000000; PREFIX=1; PREFIX_N=10000000
      GRAPH="$BIG/freshbang_data/vamana_msturing_5m.out"
      BASE="$FB/data/msturing_100m/base.fbin"; QUERY="$FB/data/msturing_100m/query.fbin"
      GT="$BIG/freshbang_data/msturing_gt_placeholder.bin"
      WLDIR="$FB/workloads/msturing-10m";;
    deep100m)
      D=96; METRIC=l2; SVF_DS=deep-fb100; BASE_OFFSET=50000000
      GRAPH="$BIG/freshbang_data/vamana_deep_50m.out"
      BASE="$FB/data/deep_100m/base.fbin"; QUERY="$FB/data/deep_10m/query.fbin"
      GT="$BIG/freshbang_data/deep_gt_placeholder.bin"
      WLDIR="$FB/workloads/deep-100m";;
    msturing100m)
      D=100; METRIC=l2; SVF_DS=msturing-fb; BASE_OFFSET=50000000
      GRAPH="$BIG/freshbang_data/vamana_msturing_50m.out"
      BASE="$FB/data/msturing_100m/base.fbin"; QUERY="$FB/data/msturing_100m/query.fbin"
      GT="$BIG/freshbang_data/msturing_gt_placeholder.bin"
      WLDIR="$FB/workloads/msturing-100m";;
    msmarco10m)
      D=769; METRIC=ip; SVF_DS=msmarco-fb; BASE_OFFSET=5000000
      GRAPH="$BIG/freshbang_data/msmarco_10m_seed.out"
      BASE="$FB/data/msmarco_10m/base.fbin"; QUERY="$FB/data/msmarco_10m/query.fbin"
      GT="$BIG/freshbang_data/msmarco_gt_placeholder.bin"
      WLDIR="$FB/workloads/msmarco-10m";;
    sift10m)
      D=128; METRIC=l2; SVF_DS=sift-fb10; BASE_OFFSET=5000000
      GRAPH="$BIG/freshbang_data/vamana_sift_5m.out"
      BASE="$BIG/freshbang_data/sift_10m_base.fbin"; QUERY="$BIG/freshbang_data/sift_query.fbin"
      GT="$BIG/freshbang_data/sift_gt_placeholder.bin"
      WLDIR="$FB/workloads/sift-10m";;
    sift100m)
      D=128; METRIC=l2; SVF_DS=sift-fb100; BASE_OFFSET=50000000
      GRAPH="$BIG/freshbang_data/vamana_sift_50m_u8.out"
      BASE="$BIG/freshbang_data/sift_100m_base.u8bin"; QUERY="$BIG/freshbang_data/sift_query.u8bin"
      GT="$BIG/freshbang_data/sift_gt_placeholder.bin"
      WLDIR="$FB/workloads/sift-100m"
      SVF_DS=sift-fb100-u8; SVF_BASE="$BIG/freshbang_data/sift_100m_base.u8bin"; SVF_QUERY="$BIG/freshbang_data/sift_query.u8bin"; SVF_DTYPE=uint8
      FB_DTYPE=uint8; FB_BASE="$BIG/freshbang_data/sift_100m_base.u8bin"; FB_QUERY="$BIG/freshbang_data/sift_query.u8bin";;
    *) echo "ERROR: unknown dataset '$1'"; exit 1;;
  esac
}
resolve "$DATASET"
RESULTS=${RESULTS:-$RESULTS_ROOT/bench_${DATASET}_$(date +%m%d_%H%M)}
WORK="$RESULTS/svf_work"
has(){ [[ ",$SYSTEMS," == *",$1,"* ]] || [ "$SYSTEMS" = all ]; }
say(){ echo -e "\n\033[1;36m== $* ==\033[0m"; }
CONDA_SH="${CONDA_SH:-/path/to/miniconda3/etc/profile.d/conda.sh}"
export CONDA_SH
activate_svf(){ set +u; source "$CONDA_SH"; conda activate svfusion; set -u; }

FDA_BASE="$BASE"
ensure_prefix(){
  [ "$PREFIX" = 1 ] || return 0
  local pdir="$BIG/bench/prefix"; mkdir -p "$pdir"
  FDA_BASE="$pdir/${DATASET}_base_${PREFIX_N}.fbin"
  if [ ! -f "$FDA_BASE" ]; then
    say "creating ${PREFIX_N}-row base prefix for $DATASET (full base large)"
    python3 "$SELFDIR/make_fbin_prefix.py" "$BASE" "$FDA_BASE" "$PREFIX_N"
  fi
}

set_D(){
  sed -i -E "s/^#define D[[:space:]].*/#define D $D/" "$1"
}

compile_fda(){  say "compile fda  (D=$D)";  set_D "$ROOT_FDA/baseline_src/vamana.h";  ( cd "$ROOT_FDA"  && make compile ) ; }
compile_fda2(){ say "compile fda2 (D=$D, P=${FDA2_P:-64}, DELLD=${FDA2_DELLD:-128}, DELK=${FDA2_DELK:-50})";  set_D "$ROOT_FDA2/baseline_src/vamana.h"; ( cd "$ROOT_FDA2" && make compile P="${FDA2_P:-64}" DELLD="${FDA2_DELLD:-128}" DELK="${FDA2_DELK:-50}" ) ; }
compile_svf(){
  [ -n "${SKIP_SVF_COMPILE:-}" ] && { echo "svf compile skipped (SKIP_SVF_COMPILE=1)"; return 0; }
  say "compile SVFusion (itopk_size=$L)"
  sed -i -E "s/(search_params\.itopk_size[[:space:]]*=[[:space:]]*)[0-9]+;/\1$L;/" "$SVF/src/workload_manager.cu"
  ( activate_svf; cd "$SVF/build" && make benchmark )
}

build_graph(){
  [ -f "$GRAPH" ] && { echo "seed graph exists: $GRAPH"; return 0; }
  say "building seed graph $GRAPH (D=$D, N=$BASE_OFFSET)"
  case "$BASE" in *.fvecs) echo "WARN: base is fvecs; builder needs fbin — convert first"; ;; esac
  set_D "$BUILDER/src/vamana.h"; ( cd "$BUILDER" && make )
  local pfx="$BIG/bench/prefix/${DATASET}_seed.fbin"; mkdir -p "$(dirname "$pfx")"
  python3 "$SELFDIR/make_fbin_prefix.py" "$BASE" "$pfx" "$BASE_OFFSET"
  "$BUILDER/bin/vamana" init.bin "$pfx" "$GRAPH" --alpha 1.2
}

run_fda_like(){
  local lbl=$1 bin=$2 gpu=$3
  [ -x "$bin" ] || { echo "MISSING binary $bin (run -a compile)"; return 1; }
  ensure_prefix
  for f in "$WLDIR"/*.jsonl; do
    local scen; scen=$(basename "$f" .jsonl)
    [ "$WL" = all ] || matches "$scen" || continue
    say "$lbl  $scen"
    if [ "$lbl" = fda2 ]; then export FDA2_MEDOID_LAST=1; else unset FDA2_MEDOID_LAST 2>/dev/null || true; fi
    CUDA_VISIBLE_DEVICES=$gpu "$bin" "$GRAPH" "$f" "$QUERY" "$GT" \
      --searchL "$L" --k "$K" --base-data "$FDA_BASE" --metric "$METRIC" $EXTRA_FDA \
      > "$RESULTS/${lbl}_${scen}.log" 2>&1
    echo "  exit=$? -> $RESULTS/${lbl}_${scen}.log"
  done
}
matches(){ local s=$1; IFS=','; for w in $WL; do [[ "$s" == *"$w"* ]] && return 0; done; return 1; }

run_svf(){
  [ -x "$BIN_SVF" ] || { echo "MISSING $BIN_SVF (run -a compile)"; return 1; }
  mkdir -p "$WORK"
  local BR="$BIG/$SVF_DS/search_results"
  activate_svf
  for f in "$WLDIR"/*.jsonl; do
    local scen; scen=$(basename "$f" .jsonl)
    [ "$WL" = all ] || matches "$scen" || continue
    local yaml="$WORK/${scen}.yaml" gt="$WORK/${scen}_gt.npy"
    local ids="$BIG/svf_delids/${SVF_DS}_${scen}" map="$WORK/${scen}_int2base.npy"
    local bin="$WORK/${scen}_results.bin" rl="$RESULTS/svf_${scen}.run.log"
    say "svf  $scen  (convert)"
    if [[ "$scen" == *clustered* ]]; then
      python3 "$SELFDIR/v11_to_svf_clustered.py" "$f" "$SVF_DS" "$yaml" "$gt" "$K" "$ids" "$map" || continue
    else
      python3 "$SELFDIR/v11_to_svf2.py" "$f" "$SVF_DS" "$yaml" "$gt" "$K" "$ids" || continue
    fi
    rm -rf "$BR"
    say "svf  $scen  (run, itopk via build)"
    CUDA_VISIBLE_DEVICES=$GPU_SVF "$BIN_SVF" -config_file="$yaml" -gpu_id=0 -data_type=$SVF_DTYPE > "$rl" 2>&1
    cp "$BR/search_1024.bin" "$bin" 2>/dev/null || { echo "  no results bin"; continue; }
    say "svf  $scen  (score)"
    if [[ "$scen" == *clustered* ]]; then
      python3 "$SELFDIR/svf_score_clustered.py" "$gt" "$bin" "$rl" "$map" | tee "$RESULTS/svf_${scen}.score.txt"
    else
      python3 "$SELFDIR/svf_score.py" "$gt" "$bin" "$rl" | tee "$RESULTS/svf_${scen}.score.txt"
    fi
  done
}

run_freshbang(){
  [ -x "$BIN_FRESHBANG" ] || { echo "MISSING $BIN_FRESHBANG"; return 1; }
  local FB_ITOPK="${FB_ITOPK:-256}"
  local fbBase="${FB_BASE:-$BASE}" fbQuery="${FB_QUERY:-$QUERY}" fbDtypeVal="${FB_DTYPE:-float}"
  local meta mm rawd fbmetric fbdist slot bdir base_bn
  meta=$(ls "$WLDIR"/*sliding*.meta.json 2>/dev/null | head -1)
  mm=$(grep -oiE '"metric"[^,}]*' "$meta" 2>/dev/null | grep -oiE 'IP|L2|cosine' | head -1)
  rawd=$(grep -oE '"dimension"[: ]+[0-9]+' "$meta" 2>/dev/null | grep -oE '[0-9]+' | head -1)
  [ -n "$rawd" ] || rawd=$D
  case "$mm" in
    IP|ip)       fbmetric=ip;     fbdist=inner_product;;
    Cosine|cosine|COSINE) fbmetric=cosine; fbdist=inner_product;;
    *)           fbmetric=l2;     fbdist=euclidean;;
  esac
  slot=$((BASE_OFFSET*2))
  local pfxdir="$BIG/fb_prefix"; mkdir -p "$pfxdir"
  local pfx
  if [ "$fbDtypeVal" = "uint8" ]; then
    pfx="$fbBase"
  else
    pfx="$pfxdir/${DATASET}_${slot}.fbin"
    [ -f "$pfx" ] || python3 "$SELFDIR/make_prefix_robust.py" "$fbBase" "$pfx" "$slot"
  fi
  bdir=$(dirname "$pfx"); base_bn=$(basename "$pfx")
  local cdir="$RESULTS/fb_cfg"; mkdir -p "$cdir" "$BIG/fb_index"
  local cfg="$cdir/freshbang.cfg"
  { echo "data_prefix=$bdir/"; echo "index_prefix=$BIG/fb_index/"; } > "$cfg"
  local dmem=device; [ "$BASE_OFFSET" -ge 50000000 ] && dmem=host
  local conf="$cdir/${DATASET}_fb.json"
  cat > "$conf" <<JSON
{ "dataset": { "name":"${DATASET}_fb", "base_file":"$base_bn", "query_file":"$base_bn", "insert_file":"$base_bn",
    "rows":$slot, "dims":$rawd, "distance":"$fbdist" },
  "search_basic_param": { "k":$K, "batch_size":10000, "total_queries":100000 },
  "index":[ { "algo":"cuvs_cagra",
    "build_param":{ "graph_degree":32, "intermediate_graph_degree":64, "graph_build_algo":"NN_DESCENT" },
    "insert_param":{ "persist":true, "refine_ratio":1.0, "graph_mem":"device", "dataset_mem":"$dmem" },
    "name":"freshbang_${DATASET}", "file":"freshbang_${DATASET}_indexfile",
    "search_params":[ { "itopk":$FB_ITOPK, "search_width":1 } ] } ] }
JSON
  echo "[freshbang] $DATASET meta-metric=$mm -> --metric $fbmetric dist=$fbdist dim=$rawd base-rows=$BASE_OFFSET slot=$slot"
  local fbMode=(); local fbDtype=(--dtype "$fbDtypeVal")
  if true; then
    say "freshbang  BUILD (once, $DATASET, dtype=$fbDtypeVal, itopk=$FB_ITOPK)"
    LD_LIBRARY_PATH="$FB_LDPATH:${LD_LIBRARY_PATH:-}" CUDA_VISIBLE_DEVICES=$GPU_FRESHBANG \
      timeout 9000 "$BIN_FRESHBANG" "$pfx" "$fbQuery" "$(ls "$WLDIR"/*.jsonl | head -1)" "$conf" --cfg "$cfg" \
      --k "$K" --base-rows "$BASE_OFFSET" --slot-rows "$slot" --metric "$fbmetric" --build "${fbDtype[@]}" \
      > "$RESULTS/freshbang_build.log" 2>&1
    local build_ec=$?
    echo "freshbang build exit=$build_ec" | tee "$RESULTS/freshbang_build.score.txt"
    [ $build_ec -eq 0 ] || { echo "freshbang build FAILED, aborting workload loop"; return 1; }
    fbMode=(--load)
  fi
  for f in "$WLDIR"/*.jsonl; do
    local scen; scen=$(basename "$f" .jsonl)
    [ "$WL" = all ] || matches "$scen" || continue
    say "freshbang  $scen"
    LD_LIBRARY_PATH="$FB_LDPATH:${LD_LIBRARY_PATH:-}" CUDA_VISIBLE_DEVICES=$GPU_FRESHBANG       timeout 18000 "$BIN_FRESHBANG" "$pfx" "$fbQuery" "$f" "$conf" --cfg "$cfg"       --k "$K" --base-rows "$BASE_OFFSET" --slot-rows "$slot" --metric "$fbmetric" "${fbMode[@]}" "${fbDtype[@]}"       > "$RESULTS/freshbang_${scen}.log" 2>&1
    local ec=$? r; r=$(grep -oE "Recall@[0-9]+: [0-9.]+%" "$RESULTS/freshbang_${scen}.log" | tail -1)
    echo "freshbang $scen exit=$ec recall=${r:-NONE}" | tee "$RESULTS/freshbang_${scen}.score.txt"
  done
  [ "$fbDtypeVal" = "uint8" ] || rm -f "$pfx"
}

mkdir -p "$RESULTS"
echo "dataset=$DATASET D=$D metric=$METRIC svf=$SVF_DS L=$L  systems=$SYSTEMS action=$ACTION  wl=$WL"
echo "results -> $RESULTS"

if [[ "$ACTION" == compile || "$ACTION" == all ]]; then
  has fda  && compile_fda
  has fda2 && compile_fda2
  has svf  && compile_svf
fi
if [[ "$ACTION" == graph || "$ACTION" == all ]]; then
  has fda || has fda2 && build_graph
fi
if [[ "$ACTION" == run || "$ACTION" == all ]]; then
  has fda  && run_fda_like fda  "$BIN_FDA"  "$GPU_FDA"
  has fda2 && run_fda_like fda2 "$BIN_FDA2" "$GPU_FDA2"
  has svf  && run_svf
  has freshbang && run_freshbang
fi
say "DONE -> $RESULTS"
