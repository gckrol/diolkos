#include "utils.h"
#include <math.h>
#include <string.h>
#include <assert.h>
#include "tensor.h"

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
    temp_q8 = Tensor_create(max(dim, hidden_dim), Q8_0);
}

void matmul_Q8_0(Tensor* xoutt, Tensor* xt, Tensor* wt, int n, int d) {
    convert_into(temp_q8, xt);

    float * xout_data = data_f32(xoutt);
    int8_t * x_data = data_i8(temp_q8);
    float * x_scale = temp_q8->scale;
    int8_t * w_data = data_i8(wt);
    float * w_scale = wt->scale;

    const int GS = 32;

    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        int in = i * n;

        // Implemented as a bunch of small groups, so the compiler will
        // have an easier time vectorizing them.

        int32_t ivals[n / GS];
        for (int j = 0; j < n; j += GS) {
            int8_t *x_start = __builtin_assume_aligned(x_data + j, 32);
            int8_t *w_start = __builtin_assume_aligned(w_data + in + j, 32);

            int32_t ival = 0;
            #pragma omp simd
            for (int k = 0; k < GS; k++) {
                ival += (int32_t)x_start[k] * (int32_t)w_start[k];
            }
            ivals[j / GS] = ival;
        }
        // Gather the scales in a nice consecutive array for SIMD.
        float scales[n / GS];
        #pragma omp simd
        for (int j = 0; j < n / GS; j++) {
            scales[j] = w_scale[in / GS + j] * x_scale[j];
        }        
        float fvals[n / GS];
        #pragma omp simd
        for (int j = 0; j < n / GS; j++) {
            fvals[j] = ((float) ivals[j]);
        }
        float sum = 0.0f;
        #pragma omp simd
        for (int j = 0; j < n / GS; j++) {
            sum += fvals[j] * scales[j];
        }        
        xout_data[i] = sum;
    }
}

void matmul(Tensor* xoutt, Tensor* xt, Tensor* wt, int n, int d) {
    if (wt->type == F32) {
        matmul_f32(xoutt, xt, wt, n, d);
    } else if (wt->type == Q8_0) {
        matmul_Q8_0(xoutt, xt, wt, n, d);
    } else {
        assert(!"Unsupported type for matmul");
    }
}
