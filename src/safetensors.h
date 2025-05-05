#ifndef SAFETENSORS_H
#define SAFETENSORS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "transformer.h"
#include "tensor.h"

typedef struct {
    Tensor* rms_att_weight; // (layer, dim) rmsnorm weights #0
    Tensor* rms_ffn_weight; // (layer, dim) #1
    // weights for matmuls. note dim == n_heads * head_size
    Tensor* wq; // (layer, dim, n_heads * head_size) #2
    Tensor* wk; // (layer, dim, n_kv_heads * head_size) #3
    Tensor* wv; // (layer, dim, n_kv_heads * head_size) #4
    Tensor* wo; // (layer, n_heads * head_size, dim) #5
    // weights for ffn
    Tensor* w1; // (layer, hidden_dim, dim) #6
    Tensor* w2; // (layer, dim, hidden_dim) #7
    Tensor* w3; // (layer, hidden_dim, dim) #8
} Layer;

typedef struct Model {
    Config *config; // pointer to the config struct
    bool huggingface_rope; // true if using huggingface style RoPE

    Tensor* token_embedding_table;    // (vocab_size, dim) #0
    Layer *layers;     // array of layers
    Tensor* rms_final_weight; // (dim,) #1
    Tensor* wcls; // (vocab_size, dim) // Often tied to token_embedding_table. #2
} Model;

/**
 * Load a model.safetensors file from the given directory
 * 
 * @param dir Directory containing the model.safetensors file
 * @return Pointer to Model struct or NULL on failure
 */
Model *load_safetensors(const char* dir);

#endif // SAFETENSORS_H
