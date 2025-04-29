#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "tensor.h"

float bf16_to_float(uint16_t bf16) {
    // BF16 has the same exponent bits as FP32 but only the top 7 mantissa bits
    // To convert: put the 16 bits in the top half of a 32-bit word, clear bottom 16 bits
    uint32_t bits = ((uint32_t)bf16) << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));  // bit-level reinterpretation
    return result;
}

float f16_to_float(uint16_t f16) {
    // Extract components
    uint32_t sign = (f16 >> 15) & 0x1;
    uint32_t exponent = (f16 >> 10) & 0x1F;
    uint32_t mantissa = f16 & 0x3FF;
    
    // Special case: zero
    if (exponent == 0 && mantissa == 0) {
        return sign ? -0.0f : 0.0f;
    }
    
    // Special case: denormalized numbers
    if (exponent == 0) {
        float result = mantissa * powf(2.0f, -24.0f);
        return sign ? -result : result;
    }
    
    // Normalized number
    // Convert to F32 format: adjust exponent bias (15 -> 127) and shift mantissa
    uint32_t f32_bits = (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13);
    float result;
    memcpy(&result, &f32_bits, sizeof(result));
    
    return result;
}

int quant_size(quantization_type type) {
    switch (type) {
        case F32: return 4;
        case F16: return 2;
        case BF16: return 2;
        case Q8_0: return 1;
        default: return -1; // Unknown type
    }
}

Tensor *Tensor_create(size_t size, quantization_type type) {
    Tensor *tensor = calloc(1, sizeof(Tensor));
    tensor->type = type;
    tensor->data = calloc(size, quant_size(type));
    if (type == Q8_0) {
        tensor->scale = calloc(1, size / 32);
    }
    return tensor;
}

float get_float(Tensor *tensor, size_t index) {
    return ((float*)tensor->data)[index];
}

void set_float(Tensor *tensor, size_t index, float value) {
    ((float*)tensor->data)[index] = value;
}
