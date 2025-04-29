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

/// Special pointer type so we don't get any autocasts.
typedef struct TensorData TensorData;

typedef struct Tensor {
    quantization_type type;
    TensorData *data;
    float *scale;
} Tensor;

float bf16_to_float(uint16_t bf16);
float f16_to_float(uint16_t f16);

Tensor *Tensor_create(size_t size, quantization_type type);

void slice(Tensor *dest, Tensor *src, int start);

float *data_f32(Tensor *tensor);

#endif // TENSOR_H
