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
    float *restrict o = data_f32(ot);
    float *restrict x = data_f32(xt);
    float *restrict weight = data_f32(weightt);

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

void matmul_Q8_0(Tensor* xoutt, Tensor* xt, Tensor* wt, int n, int d) {
    Tensor *xt_q8 = convert(xt, Q8_0); // TODO: remove memory allocation.

    float *restrict xout = data_f32(xoutt);
    int8_t *restrict x = data_i8(xt_q8);
    float *restrict xs = xt_q8->scale;
    int8_t *restrict w = data_i8(wt);
    float *restrict ws = wt->scale;

    const int GS = 32;

    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        int in = i * n;

        // Implemented as a bunch of small groups, so the compiler will
        // have an easier time vectorizing them.

        int32_t ivals[n / GS];
        for (int j = 0; j < n; j += GS) {
            int32_t ival = 0;
            for (int k = 0; k < GS; k++) {
                ival += (int32_t)x[j + k] * (int32_t)w[in + j + k];
            }
            ivals[j / GS] = ival;
        }
        // Gather the scales in a nice consecutive array for SIMD.
        float scales[n / GS];
        for (int j = 0; j < n; j += GS) {
            scales[j / GS] = ws[(in + j) / GS] * xs[j / GS];
        }
        float fvals[n / GS];
        for (int j = 0; j < n / GS; j++) {
            fvals[j] = ((float) ivals[j]);
        }
        float sum = 0.0f;
        for (int j = 0; j < n / GS; j++) {
            sum += fvals[j] * scales[j];
        }        
        xout[i] = sum;
    }

    if (xt_q8 != xt) {
        Tensor_destroy(xt_q8);
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
