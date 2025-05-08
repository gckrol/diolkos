#ifndef FNV1A_H
#define FNV1A_H

#include <stdint.h>
#include <stddef.h>

// 128-bit unsigned integer type
typedef struct {
    uint64_t low;   // least significant 64 bits
    uint64_t high;  // most significant 64 bits
} uint128_t;

// Calculate a 128-bit FNV-1a hash of data
uint128_t fnv1a_128(const void *data, size_t len);

// Continue an existing FNV-1a 128-bit hash with more data
void fnv1a_128_continue(const void *data, size_t len, uint128_t *hash);

#endif // FNV1A_H