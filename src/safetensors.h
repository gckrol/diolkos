#ifndef SAFETENSORS_H
#define SAFETENSORS_H

#include <stdint.h>
#include <stddef.h>

#include "transformer.h"

typedef struct {
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float* wq; // (layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_kv_heads * head_size)
    float* wv; // (layer, dim, n_kv_heads * head_size)
    float* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
} Layer;

typedef struct Safetensors {
    int fd;            // file descriptor
    void *data;        // pointer to mmap'd data
    size_t size;       // total size of the file
    uint64_t header_size;  // size of the header in bytes
    char *header;      // pointer to the header (JSON)
    uint8_t *tensors;  // pointer to the start of tensor data
    Config *config; // pointer to the config struct

    float* token_embedding_table;    // (vocab_size, dim)
    Layer *layers;     // array of layers
    float* rms_final_weight; // (dim,)
    float* wcls; // (vocab_size, dim) // Often tied to token_embedding_table.
} Safetensors;

/**
 * Load a model.safetensors file from the given directory
 * 
 * @param dir Directory containing the model.safetensors file
 * @return Pointer to Safetensors struct or NULL on failure
 */
Safetensors *load_safetensors(const char* dir);

/**
 * Free resources associated with a Safetensors struct
 * 
 * @param st Pointer to Safetensors struct
 */
void free_safetensors(Safetensors *st);

#endif // SAFETENSORS_H