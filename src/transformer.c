#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#include "transformer.h"
#include "utils.h"
#include "safetensors.h"

void malloc_run_state(RunState* s, Config* p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = tensor_create(p->dim, F32);
    s->xb = tensor_create(p->dim, F32);
    s->xb2 = tensor_create(p->dim, F32);
    s->hb = tensor_create(p->hidden_dim, F32);
    s->hb2 = tensor_create(p->hidden_dim, F32);
    s->q = tensor_create(p->dim, F32);
    s->att = tensor_create(p->n_heads * p->seq_len, F32);
    s->logits = tensor_create(p->vocab_size, F32);

    // Using BF16 for these saves memory, and the speed is the same.
    s->key_cache = tensor_create(p->n_layers * p->seq_len * kv_dim, BF16);
    s->value_cache = tensor_create(p->n_layers * p->seq_len * kv_dim, BF16);

    s->k = tensor_create(kv_dim, F32);
    s->v = tensor_create(kv_dim, F32);
}

void build_transformer_from_safetensors(Transformer *t, const char* model_path) {
    // Load the safetensors model
    Model *st = load_safetensors(model_path);
    if (!st) {
        fprintf(stderr, "Failed to load safetensors model from %s\n", model_path);
        exit(EXIT_FAILURE);
    }
    
    // Copy the config from safetensors
    memcpy(&t->config, st->config, sizeof(Config));
    
    // Point to the safetensors structure for later use
    t->safetensors = st;
    
    // Allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
    
    // Print the config for debugging
    fprintf(stderr, "Transformer config from safetensors: dim=%d, hidden_dim=%d, n_layers=%d, n_heads=%d, n_kv_heads=%d, vocab_size=%d, seq_len=%d\n",
            t->config.dim, t->config.hidden_dim, t->config.n_layers, t->config.n_heads,
            t->config.n_kv_heads, t->config.vocab_size, t->config.seq_len);
}

Tensor* forward(Transformer* transformer, int token, int pos) {
    // a few convenience variables
    Config* p = &transformer->config;
    Model* st = transformer->safetensors;
    RunState* s = &transformer->state;
    Tensor *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    int attention_dim = p->n_heads * head_size;

    // copy the token embedding into x
    convert_slice_into(x, st->token_embedding_table, token * dim, dim);

    // forward all the layers
    for(int l = 0; l < p->n_layers; l++) {
        // access the layer weights
        Layer *layer = &st->layers[l];

        // attention rmsnorm
        rmsnorm(s->xb, x, layer->rms_att_weight, dim);

        // key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience

        // qkv matmuls for this position
        matmul(s->q, s->xb, layer->wq, dim, attention_dim);
        matmul(s->k, s->xb, layer->wk, dim, kv_dim);
        matmul(s->v, s->xb, layer->wv, dim, kv_dim);

        if (st->huggingface_rope) {
            // RoPE relative positional encoding: using complex number rotation like in lm.rs
            for (int i = 0; i < p->n_heads; i++) {
                for (int j = 0; j < head_size/2; j++) {
                    int head_dim = j * 2;
                    float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
                    float val = pos * freq;
                    float fcr = cosf(val);
                    float fci = sinf(val);
                    
                    // Calculate indices for the first and second element in each pair
                    int q_idx1 = (i * head_size) + j;
                    int q_idx2 = (i * head_size) + j + (head_size/2);
                    
                    // For query vector - apply complex number rotation
                    float q0 = data_f32(s->q)[q_idx1];
                    float q1 = data_f32(s->q)[q_idx2];
                    data_f32(s->q)[q_idx1] = q0 * fcr - q1 * fci;
                    data_f32(s->q)[q_idx2] = q0 * fci + q1 * fcr;
                    
                    // For key vector - check if this head's key part is within kv_dim
                    // This is equivalent to the "rotn" logic in the Rust code
                    if ((i*head_size) + j + (head_size/2) < kv_dim) {
                        int k_idx1 = (i * head_size) + j;
                        int k_idx2 = (i * head_size) + j + (head_size/2);
                        
                        float k0 = data_f32(s->k)[k_idx1];
                        float k1 = data_f32(s->k)[k_idx2];
                        data_f32(s->k)[k_idx1] = k0 * fcr - k1 * fci;
                        data_f32(s->k)[k_idx2] = k0 * fci + k1 * fcr;
                    }
                }
            }
        } else {
            // Normal llama.cpp, llama2.c or lm.rs rope calculation.
            for (int i = 0; i < dim; i+=2) {
                int head_dim = i % head_size;
                float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
                float val = pos * freq;
                float fcr = cosf(val);
                float fci = sinf(val);
                int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                for (int v = 0; v < rotn; v++) {
                    Tensor* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                    float v0 = data_f32(vec)[i];
                    float v1 = data_f32(vec)[i+1];
                    data_f32(vec)[i] = v0 * fcr - v1 * fci;
                    data_f32(vec)[i+1] = v0 * fci + v1 * fcr;
                }
            }            
        }

        // copy the key and value into the kv cache (quantized)
        // The `+ pos * kv_dim` is to write it in the next position.
        // convert_f32_q8_slice_into_offset(s->key_cache, s->k, 0, kv_dim, loff + pos * kv_dim);
        // convert_f32_q8_slice_into_offset(s->value_cache, s->v, 0, kv_dim, loff + pos * kv_dim);
        // convert_f32_f32_slice_into_offset(s->key_cache, s->k, 0, kv_dim, loff + pos * kv_dim);
        // convert_f32_f32_slice_into_offset(s->value_cache, s->v, 0, kv_dim, loff + pos * kv_dim);
        convert_f32_bf16_slice_into_offset(s->key_cache, s->k, 0, kv_dim, loff + pos * kv_dim);
        convert_f32_bf16_slice_into_offset(s->value_cache, s->v, 0, kv_dim, loff + pos * kv_dim);

        // multihead attention. iterate over all heads
        #pragma omp parallel for
        for (int h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float* q = data_f32(s->q) + h * head_size;
            // attention scores for this head
            float* att = data_f32(s->att) + h * p->seq_len;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                uint16_t *data = (uint16_t*)s->key_cache->data + loff + t * kv_dim + (h / kv_mul) * head_size;

                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                #pragma omp simd
                for (int i = 0; i < head_size; i++) {
                    uint32_t bits = ((uint32_t)data[i]) << 16;
                    union { uint32_t u; float f; } u = { bits };
                    score += q[i] * u.f;
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax_f32(att, pos + 1);

            // weighted sum of the values, store back into xb
            float* xb = data_f32(s->xb) + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                uint16_t *data = (uint16_t*)s->value_cache->data + loff + t * kv_dim + (h / kv_mul) * head_size;

                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                #pragma omp simd
                for (int i = 0; i < head_size; i++) {
                    uint32_t bits = ((uint32_t)data[i]) << 16;
                    union { uint32_t u; float f; } u = { bits };                    
                    xb[i] += a * u.f;
                }
            }
        }

        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, layer->wo, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            data_f32(x)[i] += data_f32(s->xb2)[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, layer->rms_ffn_weight, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, layer->w1, dim, hidden_dim);
        matmul(s->hb2, s->xb, layer->w3, dim, hidden_dim);

        // SwiGLU non-linearity
        float *hb_data = data_f32(s->hb);
        float *hb2_data = data_f32(s->hb2);
        #pragma omp simd
        for (int i = 0; i < hidden_dim; i++) {
            float val = hb_data[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= hb2_data[i];
            hb_data[i] = val;
        }

        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, layer->w2, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            data_f32(x)[i] += data_f32(s->xb)[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, st->rms_final_weight, dim);

    // classifier into logits
    matmul(s->logits, x, st->wcls, p->dim, p->vocab_size);
    return s->logits;
}
