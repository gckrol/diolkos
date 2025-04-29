#include "utils.h"
#include <math.h>
#include <string.h>

void softmax(float* x, int size) {
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

float* rmsnorm(float* o, float* x, float* weight, int size) {
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
    return o;
}

void matmul(float* xout, float* x, float* w, int n, int d) {
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

void matmul_permuted(float* xout, float* x, float* w, int n, int d, int n_heads) {
    // For permuted weights from HuggingFace format
    // Original permutation: reshape(n_heads, dim1/n_heads/2, 2, dim2).transpose(1,2)
    
    int head_dim = d / n_heads;  // dimension per head
    int block_size = head_dim / 2;  // half of the head dimension
    
    #pragma omp parallel for
    for (int h = 0; h < n_heads; h++) {
        for (int i = 0; i < head_dim; i++) {
            float val = 0.0f;
            // Determine which block we're in (0 or 1)
            int block = i / block_size;
            // Position within the block
            int pos = i % block_size;
            
            // Calculate the offset in the weight matrix accounting for permutation
            int row_offset = h * head_dim + block * block_size + pos;
            
            for (int j = 0; j < n; j++) {
                // Access weights with the permuted indexing
                val += w[row_offset * n + j] * x[j];
            }
            
            // Store result
            xout[h * head_dim + i] = val;
        }
    }
}