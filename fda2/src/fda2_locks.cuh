#ifndef FDA2_LOCKS_CUH
#define FDA2_LOCKS_CUH

#include <cuda_runtime.h>

namespace fda2 {

// ============================================================================
// Per-node uint8 binary lock (fine-grained locking).
//
// In the sequential model, INSERT / DELETE / SEARCH never run concurrently
// with each other. Within a phase, same-type operations DO run in parallel:
//   - INSERT reverse-edge append uses atomics (no lock needed; see fda2_insert).
//   - DELETE consolidation (IP-DiskANN Algo 5) has multiple thread-blocks
//     potentially rewriting the SAME in-neighbor's adjacency list, so it takes
//     this per-node lock around the read-modify-write.
//
// CUDA atomicCAS is 32-bit only, so the byte CAS reads the containing word,
// patches the target byte, and atomicCAS's the word. Retry on contention.
//   0 = UNLOCKED, 1 = LOCKED.
// ============================================================================

__device__ __forceinline__
uint8_t atomicCAS_u8(uint8_t* p, uint8_t cmp, uint8_t val) {
    uintptr_t addr   = (uintptr_t)p;
    uint32_t* word_p = (uint32_t*)(addr & ~uintptr_t(3));
    unsigned shift   = ((unsigned)(addr & 3)) << 3;          // 0, 8, 16, or 24
    uint32_t mask    = 0xffu << shift;

    uint32_t old = *word_p;
    while (true) {
        uint8_t old_byte = (uint8_t)((old >> shift) & 0xffu);
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

    uint32_t old = *word_p;
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

} // namespace fda2

#endif // FDA2_LOCKS_CUH
