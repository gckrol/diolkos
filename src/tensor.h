#ifndef TENSOR_H
#define TENSOR_H

#include <stdint.h>
#include <stddef.h>

typedef enum quantization_type {
    F32,
    F16,
    BF16,
    Q8_0, // Group size 32.
} quantization_type;

typedef struct Tensor {
    quantization_type type;
    void *data;
    float *scale;
} Tensor;

float bf16_to_float(uint16_t bf16);
float f16_to_float(uint16_t f16);

Tensor *Tensor_create(size_t size, quantization_type type);

float *data_f32(Tensor *tensor);

#endif // TENSOR_H
