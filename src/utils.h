#ifndef UTILS_H
#define UTILS_H

#include <stdbool.h>
#include <stddef.h>

typedef struct Tensor Tensor;

// Common mathematical functions used across different components
bool reliable_isnan(double x);

size_t round_down_32(size_t i);

void softmax(Tensor* x, int size);
void softmax_f32(float* x, int size);
void rmsnorm(Tensor* o, Tensor* x, Tensor* weight, int size);
void matmul(Tensor* xout, Tensor* x, Tensor* w, int n, int d);
void matmul_permuted(Tensor* xout, Tensor* x, Tensor* w, int n, int d, int n_heads);

/// Allocate scratch memory.
void init_utils(int dim, int hidden_dim);

#endif // UTILS_H
