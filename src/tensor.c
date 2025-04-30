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
    union { uint32_t u; float f; } u = { bits };
    return u.f;
}

float f16_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h & 0x7C00) >> 10;
    uint32_t mant = (h & 0x03FF);

    uint32_t f;

    if (exp == 0) {
        // Subnormal FP16 → normalized FP32
        if (mant == 0) {
            f = sign;  // ±0
        } else {
            // Normalize subnormal
            exp = 1;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FF;
            f = sign | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        // Inf/NaN
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        // Normal number
        f = sign | ((exp + 112) << 23) | (mant << 13);
    }

    union { uint32_t u; float f; } u = { f };
    return u.f;
}

int quant_size(quantization_type type) {
    switch (type) {
        case F32: return 4;
        case F16: return 2;
        case BF16: return 2;
        case Q8_0: return 1;
        default: assert(!"unknown type");
    }
    __builtin_unreachable();
}

int group_size(quantization_type type) {
    switch (type) {
        case Q8_0: return 32;
        default: assert(!"no group size");
    }
    __builtin_unreachable();
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

float get_bf16(Tensor *tensor, size_t i) {
    assert(tensor->type == BF16);
    return bf16_to_float(((uint16_t*)tensor->data)[i]);
}

int8_t *data_i8(Tensor *tensor) {
    assert(tensor->type == Q8_0);
    return (int8_t*)tensor->data;
}

// Would be nice in the future.
// _Float16 *data_f16(Tensor *tensor) {
//     assert(tensor->type == F16);
//     return (_Float16*)tensor->data;
// }

Tensor *convert_f32_q8_0(Tensor *input) {
    assert(input->type == F32);
    Tensor *result = Tensor_create(input->dim, Q8_0);
    const int GS = 32;
    float Q_MAX = 127.0f;

    float *input_data = data_f32(input);
    int8_t *output_data = data_i8(result);

    size_t i;
    #pragma omp parallel for private(i)
    for (i = 0; i < input->dim; i += GS) {
        float max_val = 0.0f;
        #pragma omp simd
        for (size_t j = 0; j < GS; j++) {
            max_val = fmaxf(max_val, fabsf(input_data[i + j]));
        }        
        max_val /= Q_MAX;
        result->scale[i / GS] = max_val;

        #pragma omp simd
        for (size_t j = 0; j < GS; j++) {
            output_data[i + j] = (int8_t) roundf(input_data[i + j] / max_val);
        }
    }

    return result;
}

Tensor *convert_f16_q8_0(Tensor *input) {
    assert(input->type == F16);
    Tensor *result = Tensor_create(input->dim, Q8_0);
    const int GS = 32;
    float Q_MAX = 127.0f;

    int8_t *data_result = data_i8(result);

    size_t i;
    #pragma omp parallel for private(i)
    for (i = 0; i < input->dim; i += GS) {
        float floats[GS];
        for (size_t j = 0; j < GS; ++j) {
            floats[j] = get_f16(input, i + j);
        }

        float max_val = 0.0f;
        #pragma omp simd
        for (size_t j = 0; j < GS; ++j) {
            float val = floats[j];
            max_val = fmaxf(max_val, fabsf(val));
        }
        max_val /= Q_MAX;
        float inv_max = 1.0f / max_val;
        result->scale[i / GS] = max_val;

        for (size_t j = 0; j < GS; ++j) {
            data_result[i + j] = (int8_t) roundf(floats[j] * inv_max);
        }
    }

    return result;
}

Tensor *convert_bf16_q8_0(Tensor *input) {
    assert(input->type == BF16);
    Tensor *result = Tensor_create(input->dim, Q8_0);
    const int GS = 32;
    float Q_MAX = 127.0f;

    int8_t *data_result = data_i8(result);

    size_t i;
    #pragma omp parallel for private(i)
    for (i = 0; i < input->dim; i += GS) {
        float floats[GS];
        for (size_t j = 0; j < GS; ++j) {
            floats[j] = get_bf16(input, i + j);
        }

        float max_val = 0.0f;
        #pragma omp simd
        for (size_t j = 0; j < GS; ++j) {
            float val = floats[j];
            max_val = fmaxf(max_val, fabsf(val));
        }
        max_val /= Q_MAX;
        float inv_max = 1.0f / max_val;
        result->scale[i / GS] = max_val;

        for (size_t j = 0; j < GS; ++j) {
            data_result[i + j] = (int8_t) roundf(floats[j] * inv_max);
        }
    }

    return result;
}

Tensor *convert_f16_f32(Tensor *input) {
    assert(input->type == F16);
    Tensor *result = Tensor_create(input->dim, F32);
    float *data_result = data_f32(result);

    size_t i;
    #pragma omp parallel for private(i)
    for (i = 0; i < input->dim; ++i) {
        data_result[i] = get_f16(input, i);
    }

    return result;
}

Tensor *convert_bf16_f32(Tensor *input) {
    assert(input->type == BF16);
    Tensor *result = Tensor_create(input->dim, F32);
    float *data_result = data_f32(result);

    size_t i;
    #pragma omp parallel for private(i)
    for (i = 0; i < input->dim; ++i) {
        data_result[i] = get_bf16(input, i);
    }

    return result;
}

Tensor *convert(Tensor *input, quantization_type type) {
    if (input->type == type) {
        return input;
    } else if (input->type == F32 && type == Q8_0) {
        return convert_f32_q8_0(input);
    } else if (input->type == F16 && type == Q8_0) {
        return convert_f16_q8_0(input);
    } else if (input->type == BF16 && type == Q8_0) {
        return convert_bf16_q8_0(input);
    } else if (input->type == F16 && type == F32) {
        return convert_f16_f32(input);
    } else if (input->type == BF16 && type == F32) {
        return convert_bf16_f32(input);
    }
    assert(!"unknown conversion");
    return NULL;
}
