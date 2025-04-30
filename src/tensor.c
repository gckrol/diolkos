#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>

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
        default: assert(!"unknown type");
    }
}

int group_size(quantization_type type) {
    switch (type) {
        case Q8_0: return 32;
        default: assert(!"no group size");
    }
}

Tensor *Tensor_create(size_t size, quantization_type type) {
    Tensor *tensor = calloc(1, sizeof(Tensor));
    tensor->type = type;
    tensor->dim = size;
    tensor->data = calloc(size, quant_size(type));
    if (type == Q8_0) {
        tensor->scale = calloc(size / 32, sizeof(float));
    }
    return tensor;
}

void Tensor_destroy(Tensor *tensor) {
    free(tensor->data);
    free(tensor->scale);
    free(tensor);
}

void slice(Tensor *dest, Tensor *src, int start) {
    dest->type = src->type;
    dest->data = (TensorData*)((char*)src->data + start * quant_size(src->type));
    dest->scale = src->scale ? (float*)src->scale + start / 32 : NULL; // TODO group size for quant.
}

float *data_f32(Tensor *tensor) {
    assert(tensor->type == F32);
    return (float*)tensor->data;
}

float get_f16(Tensor *tensor, size_t i) {
    assert(tensor->type == F16);
    return f16_to_float(((uint16_t*)tensor->data)[i]);
}

int8_t *data_i8(Tensor *tensor) {
    assert(tensor->type == Q8_0);
    return (int8_t*)tensor->data;
}

Tensor *convert_f32_q8_0(Tensor *input) {
    assert(input->type == F32);
    Tensor *result = Tensor_create(input->dim, Q8_0);
    const int group_size = 32;
    float Q_MAX = 127.0f;

    for (size_t i = 0; i < input->dim; i += group_size) {
        float max_val = 0.0f;
        for (size_t j = 0; j < group_size; ++j) {
            float val = data_f32(input)[i + j];
            if (fabsf(val) > max_val) {
                max_val = fabsf(val);
            }
        }
        max_val /= Q_MAX;
        result->scale[i / group_size] = max_val;

        for (size_t j = 0; j < group_size; ++j) {
            data_i8(result)[i + j] = (int8_t) round(data_f32(input)[i + j] / max_val);
        }
    }

    return result;
}

Tensor *convert_f16_q8_0(Tensor *input) {
    assert(input->type == F16);
    Tensor *result = Tensor_create(input->dim, Q8_0);
    const int group_size = 32;
    float Q_MAX = 127.0f;

    for (size_t i = 0; i < input->dim; i += group_size) {
        float max_val = 0.0f;
        for (size_t j = 0; j < group_size; ++j) {
            float val = get_f16(input, i + j);
            if (fabsf(val) > max_val) {
                max_val = fabsf(val);
            }
        }
        max_val /= Q_MAX;
        result->scale[i / group_size] = max_val;

        for (size_t j = 0; j < group_size; ++j) {
            // nearbyint seems to be a bit faster than round.
            data_i8(result)[i + j] = (int8_t) nearbyint(get_f16(input, i + j) / max_val);
        }
    }

    return result;
}

Tensor *convert_f16_f32(Tensor *input) {
    assert(input->type == F16);
    Tensor *result = Tensor_create(input->dim, F32);
    float *data_result = data_f32(result);
    for (size_t i = 0; i < input->dim; ++i) {
        data_result[i] = get_f16(input, i);
    }

    return result;
}
