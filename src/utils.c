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
    float *o = data_f32(ot);
    float *x = data_f32(xt);
    float *weight = data_f32(weightt);

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
    float *xout = data_f32(xoutt);
    float *x_float = data_f32(xt);
    int8_t *w = data_i8(wt);
    float *ws = wt->scales;

    const int GS = 32;
    const float Q_MAX = 127.0f;

    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {

        float val = 0.0f;
        int in = i * n;

        // do the matmul in groups of GS
        int j;
        for (j = 0; j <= n - GS; j += GS) {

            //////
            // We should be able to quantize a block of x here.
            // TODO: we re-quantize the same block d times...
            // Better to quantize the entire thing at once before.

            float max_val = 0.0f;
            for (size_t k = 0; k < GS; k++) {
                float val = x_float[j + k];
                if (fabsf(val) > max_val) {
                    max_val = fabsf(val);
                }
            }
            max_val /= Q_MAX;
            int8_t x[32];
            for (size_t k = 0; k < GS; k++) {
                x[k] = (int8_t) nearbyint(x_float[j + k] / max_val);
            }                   

            ////
            int32_t ival = 0;
            for (int k = 0; k < GS; k++) {
                ival += ((int32_t) x[k]) * ((int32_t) w[in + j + k]);
            }
            val += ((float) ival) * ws[(in + j) / GS] * max_val;
        }

        xout[i] = val;
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
