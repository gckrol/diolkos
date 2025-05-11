#ifndef THREADING_H
#define THREADING_H

typedef struct Tensor Tensor;

void matmul_parallel(Tensor* out_tensor, Tensor* in_tensor, Tensor* matrix);
void matmul_parallel_f32(Tensor* out_tensor, Tensor* in_tensor, Tensor* matrix);
void init_threads();

#endif // THREADING_H
