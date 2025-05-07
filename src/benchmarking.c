#include "benchmarking.h"
#include "tensor.h"
#include "utils.h"
#include <stdlib.h>
#include <assert.h>

static Tensor* input;
static Tensor* output;
static Tensor* matrix;

static inline float fast_random(uint32_t *state) {
    *state = *state * 1664525 + 1013904223; // LCG
    return (float)(*state & 0x7FFFFFFF) / (float)0x40000000 - 1.0f;
}

void benchmark_init(int input_size, int output_size, quant_t type) {
    assert(type == Q8_0);

    size_t matrix_size = (size_t)input_size * output_size;
    input = tensor_create(input_size, F32);
    output = tensor_create(output_size, F32);
    matrix = tensor_create(matrix_size, type);

    // Fill the input and matrix.
    uint32_t rng_state = 42;
    float *input_data = data_f32(input);
    for (int i = 0; i < input_size; i++) {
        input_data[i] = fast_random(&rng_state);
    }
    int8_t *matrix_data = data_i8(matrix);
    for (int i = 0; i < matrix_size; i++) {
        fast_random(&rng_state);
        matrix_data[i] = rng_state & 0xFF - 0x80;
    }
    for (int i = 0; i < matrix_size/32; i++) {
        matrix->scale[i] = fast_random(&rng_state);
    }
}

void benchmark_destroy(void) {
    tensor_destroy(input);
    tensor_destroy(output);
    tensor_destroy(matrix);
}

void benchmark_run(void) {
    matmul(output, input, matrix, input->dim, output->dim);
}
