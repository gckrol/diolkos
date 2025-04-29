#ifndef UTILS_H
#define UTILS_H

typedef struct Tensor Tensor;

// Common mathematical functions used across different components
void softmax(Tensor* x, int size);
void softmax_f32(float* x, int size);
void rmsnorm(Tensor* o, Tensor* x, Tensor* weight, int size);
void matmul(Tensor* xout, Tensor* x, Tensor* w, int n, int d);
void matmul_permuted(Tensor* xout, Tensor* x, Tensor* w, int n, int d, int n_heads);

#endif // UTILS_H
