#include "utils.h"
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>

#include "tensor.h"

bool reliable_isnan(double x) {
    uint64_t bits;
    memcpy(&bits, &x, sizeof(bits));
    uint64_t exponent = (bits >> 52) & 0x7FF;
    uint64_t mantissa = bits & ((1ULL << 52) - 1);
    return (exponent == 0x7FF) && (mantissa != 0);
}

void softmax(Tensor* xt, int size) {
    float *x = data_f32(xt);
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void softmax_f32(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void rmsnorm(Tensor* ot, Tensor* xt, Tensor* weightt, int size) {
    float * o = data_f32(ot);
    float * x = data_f32(xt);
    float * weight = data_f32(weightt);

    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void matmul_f32(Tensor* xoutt, Tensor* xt, Tensor* wt, int n, int d) {
    float *xout = data_f32(xoutt);
    float *x = data_f32(xt);
    float *w = data_f32(wt);

    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;

        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

static int max(int a, int b) {
    return (a > b) ? a : b;
}   

Tensor *temp_q8 = NULL;
void init_utils(int dim, int hidden_dim) {
    temp_q8 = tensor_create(max(dim, hidden_dim), Q8_0);
}

void matmul_Q8_0(Tensor* out_tensor, Tensor* in_tensor, Tensor* matrix, int in_dim, int out_dim) {
    convert_into(temp_q8, in_tensor);

    float * xout_data = __builtin_assume_aligned(data_f32(out_tensor), 32);
    int8_t * x_data = __builtin_assume_aligned(data_i8(temp_q8), 32);
    float * x_scale = __builtin_assume_aligned(temp_q8->scale, 32);
    int8_t * w_data = __builtin_assume_aligned(data_i8(matrix), 32);
    float * w_scale = __builtin_assume_aligned(matrix->scale, 32);

    const int GS = 32;

    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < out_dim; i++) {
        int in = i * in_dim;

        float sum = 0.0f;
        for (int j = 0; j < in_dim / GS; j++) {
            int8_t *x_start = __builtin_assume_aligned(x_data + j * GS, 32);
            int8_t *w_start = __builtin_assume_aligned(w_data + in + j * GS, 32);

            int32_t ival = 0;
            #pragma omp simd simdlen(16)
            for (int k = 0; k < GS; k++) {
                ival += (int32_t)x_start[k] * (int32_t)w_start[k];
            }
            // It's faster to do this right away, in this loop.
            // This might be because we can do the multiplication while waiting for memory.
            sum += ((float) ival) * w_scale[in / GS + j] * x_scale[j];
        }        
        xout_data[i] = sum;
    }
}

void matmul_Q4_0_untested(Tensor* xoutt, Tensor* xt, Tensor* wt, int n, int d) {
    convert_into(temp_q8, xt);

    float * xout_data = __builtin_assume_aligned(data_f32(xoutt), 32);
    int8_t * x_data = __builtin_assume_aligned(data_i8(temp_q8), 32);
    float * x_scale = __builtin_assume_aligned(temp_q8->scale, 32);
    int8_t * w_data = __builtin_assume_aligned(data_i8(wt), 32);
    float * w_scale = __builtin_assume_aligned(wt->scale, 32);

    const int GS = 32;

    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        int in = i * n;

        float sum = 0.0f;
        for (int j = 0; j < n / GS; j++) {
            int8_t *x_start = __builtin_assume_aligned(x_data + j * GS, 32); // Q8
            int8_t *w_start = __builtin_assume_aligned(w_data + in + j * GS / 2, 32); // Q4

            // Unpack w into Q8
            int8_t w_start_unpacked[GS] __attribute__((aligned(32)));
            #pragma omp simd simdlen(16)
            for (int k = 0; k < GS; k+=2) {
                int8_t w1 = (w_start[k/2] << 4) >> 4;
                int8_t w2 = w_start[k/2] >> 4;

                w_start_unpacked[k] = w1;
                w_start_unpacked[k+1] = w2;
            }

            // Q8 kernel
            int32_t ival = 0;
            #pragma omp simd simdlen(16)
            for (int k = 0; k < GS; k++) {
                ival += (int32_t)x_start[k] * (int32_t)w_start_unpacked[k];
            }

            sum += ((float)(ival)) * w_scale[in / GS + j] * x_scale[j];
        }        
        xout_data[i] = sum;
    }
}

void matmul(Tensor* xoutt, Tensor* xt, Tensor* wt, int n, int d) {
    assert(wt->dim >= n * d);
    assert(xt->dim >= n);
    assert(xoutt->dim >= d);
    if (wt->type == F32) {
        matmul_f32(xoutt, xt, wt, n, d);
    } else if (wt->type == Q8_0) {
        matmul_Q8_0(xoutt, xt, wt, n, d);
    } else if (wt->type == Q4_0) {
        assert(!"Q4_0 matmul is untested!");
        matmul_Q4_0_untested(xoutt, xt, wt, n, d);
    } else {
        assert(!"Unsupported type for matmul");
    }
}

size_t round_down_32(size_t i) {
   return i - (i % 32);
}
