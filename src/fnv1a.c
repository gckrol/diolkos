#include "fnv1a.h"

// FNV-1a 32-bit constants
#define FNV_PRIME_32 16777619U
#define FNV_OFFSET_32 2166136261U

uint128_t fnv1a_128(const void *data, size_t len) {
    // Initialize with consistent values to match original hash behavior
    uint128_t hash = { .high = 0x6c62272e07bb0142ULL, .low = 0x62b821756295c58dULL };
    fnv1a_128_continue(data, len, &hash);
    return hash;
}

void fnv1a_128_continue(const void *data, size_t len, uint128_t *hash) {
    const uint32_t *words = (const uint32_t *)data;
    size_t word_count = len / 4;
    const uint8_t *tail_bytes = (const uint8_t *)data;

    uint32_t h[8];
    h[0] = (uint32_t)(hash->low & 0xFFFFFFFFULL);
    h[1] = (uint32_t)(hash->low >> 32);
    h[2] = (uint32_t)(hash->high & 0xFFFFFFFFULL);
    h[3] = (uint32_t)(hash->high >> 32);
    h[4] = h[0];
    h[5] = h[1];
    h[6] = h[2];
    h[7] = h[3];

    size_t i = 0;

    // Process 8 × 32-bit words at a time
    for (; i + 8 <= word_count; i += 8) {
        #pragma omp simd
        for (int j = 0; j < 8; j++) {
            h[j] ^= words[i + j];
            h[j] *= FNV_PRIME_32;
        }
    }

    // Handle remaining 32-bit words
    for (; i < word_count; i++) {
        h[i % 8] ^= words[i];
        h[i % 8] *= FNV_PRIME_32;
    }

    // Handle remaining bytes
    size_t byte_start = word_count * 4;
    for (; byte_start < len; byte_start++) {
        h[(byte_start / 4) % 8] ^= tail_bytes[byte_start];
        h[(byte_start / 4) % 8] *= FNV_PRIME_32;
    }

    // Final mixing
    for (int j = 0; j < 8; j++) {
        h[j] ^= h[j] >> 16;
        h[j] *= 0x85ebca6b;
        h[j] ^= h[j] >> 13;
    }

    // Combine into 128-bit hash
    hash->low  = ((uint64_t)h[1] << 32) | h[0];
    hash->low ^= ((uint64_t)h[5] << 32) ^ h[4];
    hash->high = ((uint64_t)h[3] << 32) | h[2];
    hash->high ^= ((uint64_t)h[7] << 32) ^ h[6];
}