# v11 workload drivers

Shared tooling for the **v11 NeurIPS-style** streaming workloads under
`${DATA_ROOT}/freshbang/workloads/<dataset>/`.

## v11 format recap

Each workload file is a sequence of pretty-printed JSON objects (one object
spans many lines). Events are **batched** and carry **IDs only** — vectors are
not embedded.

```json
{ "op_type":"insert", "step":1, "vector_ids":[500000, ...], "num_vectors":10000 }
{ "op_type":"delete", "step":3, "vector_ids":[0, ...],      "num_vectors":10000 }
{ "op_type":"search", "step":2, "point_id":[0, ...],        "groundtruth":[[...],...] }
```

- insert/delete `vector_ids` index the dataset **base** vectors.
- search `point_id` indexes the **query** set; `groundtruth[i]` is the top-k id
  list for query i, computed against the active set at that step.

Vectors live separately in `${DATA_ROOT}/freshbang/data/<dataset>/`
(`.fbin` = `[u32 n][u32 dim][floats]`, `.fvecs` = per-row `[i32 dim][floats]`,
glove is hdf5 — pre-convert to fbin).

## Files

| File | What |
|---|---|
| `v11_loader.hpp` | Header-only, dependency-free. Streaming multi-line JSON splitter + fbin/fvecs loader. Two entry points: `v11::stream` (per-item callbacks) and `v11::stream_batched` (whole-batch callbacks). |
| `v11_selftest.cpp` | Validates parse counts, id ranges, and recomputes brute-force GT for the first query batch vs the embedded GT. |
| `freshbang_v11_driver.cpp` | Drives FreshBANG on a v11 workload (loads base+query, batched insert/delete/search, recall vs embedded GT). |
| `Makefile` | `make` builds both. FreshBANG driver uses the cuvs_bench conda g++ for GLIBCXX compatibility. |

## Build

```bash
cd "$PROJECT_ROOT/freshbang/v11_drivers"
make                      # builds v11_selftest (plain g++) + freshbang_v11_driver (env g++)
```

## Validate a workload's GT (no ANNS needed)

```bash
./v11_selftest <workload.jsonl> <base> <query> <dim> <l2|cosine> <base_offset> <k>
# e.g. sift-1m:
./v11_selftest \
  "$DATA_ROOT/freshbang/workloads/sift-1m/sift-1m_sliding_window.jsonl" \
  "$DATA_ROOT/freshbang/data/sift_1m/base.fvecs" \
  "$DATA_ROOT/freshbang/data/sift_1m/query.fvecs" \
  128 l2 500000 100
# -> brute-force top-100 vs embedded top-100: overlap 100/100 (exact)
```

## Run FreshBANG on a v11 workload

```bash
ENV=/mnt/ssd_volume/mumba1/envs/cuvs_bench_26_02
LD_LIBRARY_PATH=/tmp/freshbang:$ENV/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 \
./freshbang_v11_driver \
  "$DATA_ROOT/freshbang/data/sift_1m/base.fvecs" \
  "$DATA_ROOT/freshbang/data/sift_1m/query.fvecs" \
  "$DATA_ROOT/freshbang/workloads/sift-1m/sift-1m_sliding_window.jsonl" \
  <freshbang_conf.json> --cfg <freshbang.cfg> \
  --k 10 --base-rows 500000 --slot-rows 1000000 --metric l2 [--max-cycles N]
```

`--metric cosine` L2-normalizes base+query and uses inner-product (cosine ≡ IP
on normalized vectors) — for glove and text2image.

## Per-dataset parameters

| dataset | base / query | dim | metric | base-rows | slot-rows |
|---|---|---|---|---|---|
| sift-1m        | sift_1m/base.fvecs, query.fvecs       | 128 | l2     | 500000 | 1000000 |
| glove-100      | glove_100/base.fbin, query.fbin*      | 100 | cosine | 600000 | 1183514 |
| deep-1m        | deep_10m/base.fbin, query.fbin        | 96  | l2     | 500000 | 1000000 |
| text2image-1m  | text2image_10m/base.fbin, query.fbin  | 200 | cosine | 500000 | 1000000 |
| wikipedia-1m   | wikipedia_10m/base.fbin, query.fbin   | 768 | l2     | 500000 | 1000000 |

*glove vectors come from `glove_100/base.hdf5` (`train`/`test`) — convert once to fbin.

## Wiring into the GPU executors (MVCC / BANG-vamana / FDA-Naive)

Each executor already consumes a `std::vector<WorkloadEvent>` (one vector per
event). To feed it a v11 workload, in `main.cu`:

```cpp
#include "v11_loader.hpp"
v11::VecStore base = v11::loadVectors(baseDataPath, slotRows);
std::vector<WorkloadEvent> events;
v11::stream(workloadPath,
  [&](unsigned id){ WorkloadEvent e; e.type=EVENT_INSERT; e.pointId=id;
                    e.vector.assign(base.row(id), base.row(id)+base.dim);
                    events.push_back(std::move(e)); },
  [&](unsigned id){ WorkloadEvent e; e.type=EVENT_DELETE; e.pointId=id;
                    events.push_back(std::move(e)); },
  [&](unsigned qid, const std::vector<unsigned>& gt){ WorkloadEvent e;
                    e.type=EVENT_QUERY; e.queryId=qid; e.groundtruth=gt;
                    events.push_back(std::move(e)); });
```

Query vectors keep coming from the executor's own `queries.bin` (indexed by
`queryId`), unchanged. Note each executor has a compile-time `D`, so rebuild per
dataset dimension (128/100/96/200/768).
