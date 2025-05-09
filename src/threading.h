#ifndef THREADING_H
#define THREADING_H

typedef struct Tensor Tensor;

void matmul_parallel(Tensor* out_tensor, Tensor* in_tensor, Tensor* matrix, int in_dim, int out_dim);
void init_threads();

#endif // THREADING_H
