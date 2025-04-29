#ifndef SAFETENSORS_H
#define SAFETENSORS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "transformer.h"
#include "tensor.h"

typedef struct {
    Tensor* rms_att_weight; // (layer, dim) rmsnorm weights
    Tensor* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    Tensor* wq; // (layer, dim, n_heads * head_size)
    Tensor* wk; // (layer, dim, n_kv_heads * head_size)
    Tensor* wv; // (layer, dim, n_kv_heads * head_size)
    Tensor* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    Tensor* w1; // (layer, hidden_dim, dim)
    Tensor* w2; // (layer, dim, hidden_dim)
    Tensor* w3; // (layer, hidden_dim, dim)
} Layer;

typedef struct Model {
    Config *config; // pointer to the config struct
    bool huggingface_rope; // true if using huggingface style RoPE

    Tensor* token_embedding_table;    // (vocab_size, dim)
    Layer *layers;     // array of layers
    Tensor* rms_final_weight; // (dim,)
    Tensor* wcls; // (vocab_size, dim) // Often tied to token_embedding_table.
} Model;

/**
 * Load a model.safetensors file from the given directory
 * 
 * @param dir Directory containing the model.safetensors file
 * @return Pointer to Model struct or NULL on failure
 */
Model *load_safetensors(const char* dir);

#endif // SAFETENSORS_H
