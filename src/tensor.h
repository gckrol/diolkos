#ifndef TENSOR_H
#define TENSOR_H

#include <stdint.h>
#include <stddef.h>

typedef enum quant_t {
    F32,
    F16,
    BF16,
    Q8_0, // Group size 32.
    Q4_0, // Group size 32.
} quant_t;

/// Special pointer type so we don't get any autocasts.
typedef struct TensorData TensorData;

typedef struct Tensor {
    quant_t type;
    TensorData *data;
    float *scale;
    size_t dim;
    size_t hdim; // Input dimension.
    size_t vdim; // Output dimension.

    uint32_t tensor_id; // Used to identify it on workers.
    int fd;
} Tensor;

float bf16_to_float(uint16_t bf16);
float f16_to_float(uint16_t f16);

Tensor *tensor_create(size_t size, quant_t type);
void tensor_destroy(Tensor *tensor);
size_t Tensor_storage_size(Tensor *tensor);
void tensor_validate(Tensor *tensor);

Tensor *convert(Tensor *input, quant_t type);
void convert_into(Tensor *dst, Tensor *input);
void convert_slice_into(Tensor *dst, Tensor *input, size_t start, size_t length);
void convert_f32_q8_slice_into_offset(Tensor *dst, Tensor *input, size_t start, size_t length, size_t dst_offset);
void convert_f32_f32_slice_into_offset(Tensor *dst, Tensor *input, size_t start, size_t length, size_t dst_offset);
void convert_bf16_f32_slice_into_offset(Tensor *dst, Tensor *input, size_t start, size_t length, size_t dst_offset);
void convert_f32_bf16_slice_into_offset(Tensor *dst, Tensor *input, size_t start, size_t length, size_t dst_offset);
void convert_q8_f32_slice_into_offset(Tensor *dst, Tensor *input, size_t start, size_t length, size_t offset);
    
Tensor *convert_f32_q8(Tensor *input);
Tensor *convert_f16_q8_0(Tensor *input);
Tensor *convert_f16_f32(Tensor *input);

void slice(Tensor *dest, Tensor *src, int start);

float *data_f32(Tensor *tensor);
int8_t *data_i8(Tensor *tensor);

int group_size(quant_t type);
int quant_size(quant_t type);

size_t tensor_memory(Tensor *tensor);

const char* quant_t_to_string(quant_t type);

#endif // TENSOR_H
