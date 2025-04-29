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

// Utility function for safe allocation with error handling
void* safe_calloc(size_t count, size_t size, const char* description) {
    void* ptr = calloc(count, size);
    if (ptr == NULL) {
        fprintf(stderr, "Failed to allocate memory for %s (%zu bytes)\n", 
                description, count * size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void malloc_run_state(RunState* s, Config* p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = safe_calloc(p->dim, sizeof(float), "RunState x");
    s->xb = safe_calloc(p->dim, sizeof(float), "RunState xb");
    s->xb2 = safe_calloc(p->dim, sizeof(float), "RunState xb2");
    s->hb = safe_calloc(p->hidden_dim, sizeof(float), "RunState hb");
    s->hb2 = safe_calloc(p->hidden_dim, sizeof(float), "RunState hb2");
    s->q = safe_calloc(p->dim, sizeof(float), "RunState q");
    s->key_cache = safe_calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float), "RunState key_cache");
    s->value_cache = safe_calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float), "RunState value_cache");
    s->att = safe_calloc(p->n_heads * p->seq_len, sizeof(float), "RunState att");
    s->logits = safe_calloc(p->vocab_size, sizeof(float), "RunState logits");
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

void build_transformer_from_safetensors(Transformer *t, const char* model_path) {
    // Load the safetensors model
    Safetensors *st = load_safetensors(model_path);
    if (!st) {
        fprintf(stderr, "Failed to load safetensors model from %s\n", model_path);
        exit(EXIT_FAILURE);
    }
    
    // Copy the config from safetensors
    memcpy(&t->config, st->config, sizeof(Config));
    
    // Set up the transformer weights mapping to safetensors data
    t->weights.token_embedding_table = st->token_embedding_table;
    t->weights.rms_final_weight = st->rms_final_weight;
    t->weights.wcls = st->wcls;
    
    // Point to the safetensors structure for later use
    t->safetensors = st;
    
    // Allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
    
    // Print the config for debugging
    fprintf(stderr, "Transformer config from safetensors: dim=%d, hidden_dim=%d, n_layers=%d, n_heads=%d, n_kv_heads=%d, vocab_size=%d, seq_len=%d\n",
            t->config.dim, t->config.hidden_dim, t->config.n_layers, t->config.n_heads,
            t->config.n_kv_heads, t->config.vocab_size, t->config.seq_len);
}

float* forward(Transformer* transformer, int token, int pos) {
    // a few convenience variables
    Config* p = &transformer->config;
    Safetensors* st = transformer->safetensors;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    int attention_dim = p->n_heads * head_size;

    // copy the token embedding into x
    float* content_row = st->token_embedding_table + token * dim;
    memcpy(x, content_row, dim*sizeof(*x));

    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {
        // access the layer weights
        Layer *layer = &st->layers[l];

        // attention rmsnorm
        rmsnorm(s->xb, x, layer->rms_att_weight, dim);

        // key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

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
                    float q0 = s->q[q_idx1];
                    float q1 = s->q[q_idx2];
                    s->q[q_idx1] = q0 * fcr - q1 * fci;
                    s->q[q_idx2] = q0 * fci + q1 * fcr;
                    
                    // For key vector - check if this head's key part is within kv_dim
                    // This is equivalent to the "rotn" logic in the Rust code
                    if ((i*head_size) + j + (head_size/2) < kv_dim) {
                        int k_idx1 = (i * head_size) + j;
                        int k_idx2 = (i * head_size) + j + (head_size/2);
                        
                        float k0 = s->k[k_idx1];
                        float k1 = s->k[k_idx2];
                        s->k[k_idx1] = k0 * fcr - k1 * fci;
                        s->k[k_idx2] = k0 * fci + k1 * fcr;
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
                    float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                    float v0 = vec[i];
                    float v1 = vec[i+1];
                    vec[i]   = v0 * fcr - v1 * fci;
                    vec[i+1] = v0 * fci + v1 * fcr;
                }
            }            
        }


        // multihead attention. iterate over all heads
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float* q = s->q + h * head_size;
            // attention scores for this head
            float* att = s->att + h * p->seq_len;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, layer->wo, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, layer->rms_ffn_weight, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, layer->w1, dim, hidden_dim);
        matmul(s->hb2, s->xb, layer->w3, dim, hidden_dim);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, layer->w2, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, st->rms_final_weight, dim);

    // classifier into logits
    matmul(s->logits, x, st->wcls, p->dim, p->vocab_size);
    return s->logits;
}