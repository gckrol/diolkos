#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#include "tensor.h"

// Forward declaration for Model struct
typedef struct Model Model;
typedef struct Tensor Tensor;

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;


typedef struct {
    // current wave of activations
    Tensor *x; // activation at current time stamp (dim,)
    Tensor *xb; // same, but inside a residual branch (dim,)
    Tensor *xb2; // an additional buffer just for convenience (dim,)
    Tensor *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    Tensor *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    Tensor *q; // query (dim,)

    // Tensor *k; // key (dim,)
    // Tensor *v; // value (dim,)
    
    Tensor *att; // buffer for scores/attention values (n_heads, seq_len)
    Tensor *logits; // output logits
    // kv cache
    Tensor* key_cache;   // (layer, seq_len, dim)
    Tensor* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct Transformer {
    Config config; // the hyperparameters of the architecture (the blueprint)
    RunState state; // buffers for the "wave" of activations in the forward pass
    Model* safetensors;
} Transformer;

// Functions for direct checkpoint loading
void malloc_run_state(RunState* s, Config* p);
void build_transformer_from_safetensors(Transformer *t, const char* model_path);
Tensor* forward(Transformer* transformer, int token, int pos);

#endif // TRANSFORMER_H
