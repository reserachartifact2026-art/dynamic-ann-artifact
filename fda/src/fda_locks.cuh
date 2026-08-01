#ifndef FDA_LOCKS_CUH
#define FDA_LOCKS_CUH

#include <cuda_runtime.h>

namespace fda {

// ============================================================================
// Per-node uint8 binary lock (FreshDiskANN paper: "uint8 binary lock per node,
// fine-grained locking"). CUDA's atomicCAS is 32-bit only, so the byte CAS is
// implemented by reading the containing 32-bit word, computing the proposed
// new word with the target byte updated, and atomicCAS'ing the word. Retry
// on contention.
//
// State convention:
//   0  = UNLOCKED
//   1  = LOCKED
// ============================================================================

__device__ __forceinline__
uint8_t atomicCAS_u8(uint8_t* p, uint8_t cmp, uint8_t val) {
    uintptr_t addr   = (uintptr_t)p;
    uint32_t* word_p = (uint32_t*)(addr & ~uintptr_t(3));
    unsigned shift   = ((unsigned)(addr & 3)) << 3;          // 0, 8, 16, or 24
    uint32_t mask    = 0xffu << shift;

    // Force fresh load each call (atomicAdd by 0 = atomic load) — prevents the
    // compiler from hoisting *word_p out of an outer spinlock loop, which
    // would cause spinners to read stale "lock=1" forever after the holder
    // released. Same pattern in bn_atomicCAS_u8 in baseline_src/greedySearch.cu.
    uint32_t old = atomicAdd(word_p, 0u);
    while (true) {
        uint8_t  old_byte = (uint8_t)((old >> shift) & 0xffu);
        if (old_byte != cmp) return old_byte;                // CAS would fail; bail
        uint32_t new_word = (old & ~mask) | ((uint32_t)val << shift);
        uint32_t prev     = atomicCAS(word_p, old, new_word);
        if (prev == old) return cmp;                          // success
        old = prev;                                            // retry with fresh word
    }
}

__device__ __forceinline__
void atomicExch_u8(uint8_t* p, uint8_t val) {
    uintptr_t addr   = (uintptr_t)p;
    uint32_t* word_p = (uint32_t*)(addr & ~uintptr_t(3));
    unsigned shift   = ((unsigned)(addr & 3)) << 3;
    uint32_t mask    = 0xffu << shift;

    uint32_t old = atomicAdd(word_p, 0u);   // force fresh load
    while (true) {
        uint32_t new_word = (old & ~mask) | ((uint32_t)val << shift);
        uint32_t prev     = atomicCAS(word_p, old, new_word);
        if (prev == old) return;
        old = prev;
    }
}

__device__ __forceinline__
void acquireNodeLock(uint8_t* d_locks, unsigned v) {
    while (atomicCAS_u8(&d_locks[v], 0u, 1u) != 0u) { /* spin */ }
    __threadfence();
}

__device__ __forceinline__
bool tryAcquireNodeLock(uint8_t* d_locks, unsigned v) {
    bool got = (atomicCAS_u8(&d_locks[v], 0u, 1u) == 0u);
    if (got) __threadfence();
    return got;
}

__device__ __forceinline__
void releaseNodeLock(uint8_t* d_locks, unsigned v) {
    __threadfence();
    atomicExch_u8(&d_locks[v], 0u);
}

} // namespace fda

#endif // FDA_LOCKS_CUH
