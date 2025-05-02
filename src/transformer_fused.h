#ifndef TRANSFORMER_FUSED_H
#define TRANSFORMER_FUSED_H

typedef struct Tensor Tensor;
typedef struct Transformer Transformer;

Tensor* forward_fused(Transformer* transformer, int token, int pos);

#endif // TRANSFORMER_FUSED_H