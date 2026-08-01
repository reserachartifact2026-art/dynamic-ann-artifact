#ifndef VAMANA_H
#include "vamana.h"
#endif

__device__ unsigned bf_hashFn1(unsigned x) {
	// FNV-1a hash
	uint64_t hash = 0xcbf29ce4;
	hash = (hash ^ (x & 0xff)) * 0x01000193;
	hash = (hash ^ ((x >> 8) & 0xff)) * 0x01000193;
	hash = (hash ^ ((x >> 16) & 0xff)) * 0x01000193;
	hash = (hash ^ ((x >> 24) & 0xff)) * 0x01000193;

	return hash % BF_ENTRIES;
}

__device__ unsigned bf_hashFn2(unsigned x) {
	// FNV-1a hash
	uint64_t hash = 0x84222325;
	hash = (hash ^ (x & 0xff)) * 0x1B3;
	hash = (hash ^ ((x >> 8) & 0xff)) * 0x1B3;
	hash = (hash ^ ((x >> 16) & 0xff)) * 0x1B3;
	hash = (hash ^ ((x >> 24) & 0xff)) * 0x1B3;
	return hash % BF_ENTRIES;
}

__device__ bool bf_check(bool *bf, unsigned x) {
    return bf[bf_hashFn1(x)] && bf[bf_hashFn2(x)];
}

__device__ void bf_set(bool *bf, unsigned x) {
    bf[bf_hashFn1(x)] = true;
    bf[bf_hashFn2(x)] = true;
}
