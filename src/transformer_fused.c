#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <assert.h>

#include "transformer.h"
#include "utils.h"
#include "safetensors.h"

static Tensor *temp_q8 = NULL;

static int max(int a, int b) {
    return (a > b) ? a : b;
}

static int min(int a, int b) {
    return (a < b) ? a : b;
}

/**
 * Forward pass, specialized for Q8 quantization, with everything fused.
 * This should help performance.
 */
Tensor* forward_fused(Transformer* transformer, int token, int pos) {
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

    const int GS = 32;
    const float Q_MAXF = 127.0f;    

    if (!temp_q8) {
        // TODO not the cleanest way to allocate, should be in runstate.
        temp_q8 = Tensor_create(max(dim, hidden_dim), Q8_0);
    }

    // Precomputed fcr and fci for RoPE
    float fcr[head_size/2];
    float fci[head_size/2];
    for (int j = 0; j < head_size/2; j++) {
        int head_dim = j * 2;
        float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
        float val = pos * freq;
        fcr[j] = cosf(val);
        fci[j] = sinf(val);
    }

    // copy the token embedding into x
    convert_slice_into(x, st->token_embedding_table, token * dim, dim);

    // forward all the layers
    for(int l = 0; l < p->n_layers; l++) {
        // access the layer weights
        Layer *layer = &st->layers[l];

        //////////////////////
        // attention rmsnorm
        // rmsnorm(s->xb, x, layer->rms_att_weight, dim);
        // void rmsnorm(Tensor* ot, Tensor* xt, Tensor* weightt, int size)

        float * xb_data = data_f32(s->xb);
        float * x_data = data_f32(x);
        float * weight_data = data_f32(layer->rms_att_weight);
    
        // calculate sum of squares
        float ss = 0.0f;
        for (int j = 0; j < s->xb->dim; j++) {
            ss += x_data[j] * x_data[j];
        }
        ss /= s->xb->dim;
        ss += 1e-5f;
        ss = 1.0f / sqrtf(ss);
        // normalize and scale (fused into quant below)
        // for (int j = 0; j < s->xb->dim; j++) {
        //     xb_data[j] = weight_data[j] * (ss * x_data[j]);
        // }        

        //////////////////////////////////
        // qkv matmuls for this position

        // key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience

        // F32, F32, Q8_0

        // Quantize xb into temp_q8 so we can do a q8 matrix multiplication.
        // convert_into(temp_q8, s->xb);
        // void convert_f32_q8_slice_into_offset(Tensor *dst, Tensor *input, size_t start, size_t length, size_t dst_offset);
    
        int8_t * temp_data = data_i8(temp_q8);
        float * temp_scale = temp_q8->scale;        
    
        size_t it;
        #pragma omp parallel for private(it)
        for (it = 0; it < s->xb->dim; it += GS) {
            float max_val = 0.0f;
            #pragma omp simd
            for (int j = 0; j < GS; j++) {
                // Normalize and scale (fused into here).
                xb_data[it + j] = weight_data[it + j] * (ss * x_data[it + j]);
                max_val = fmaxf(max_val, fabsf(xb_data[it + j]));
            }        
            max_val /= Q_MAXF;
            temp_scale[it / GS] = max_val;
    
            #pragma omp simd
            for (int j = 0; j < GS; j++) {
                temp_data[it + j] = (int8_t) roundf(xb_data[it + j] / max_val);
            }
        }

        ////

        // Fused matmuls.
        // matmul(s->q, s->xb, layer->wq, dim, attention_dim);
        // matmul(s->k, s->xb, layer->wk, dim, kv_dim);
        // matmul(s->v, s->xb, layer->wv, dim, kv_dim);

        // void matmul_Q8_0(Tensor* xoutt, Tensor* xt, Tensor* wt, int n, int d)

        float * q_data = data_f32(s->q);
        int8_t * wq_data = data_i8(layer->wq);
        float * wq_scale = layer->wq->scale;
    
        int i;
        #pragma omp parallel for private(i)
        for (i = 0; i < attention_dim; i++) {
            int in = i * dim;
    
            // Implemented as a bunch of small groups, so the compiler will
            // have an easier time vectorizing them.
    
            int32_t ivals_q[dim / GS];
            for (int j = 0; j < dim; j += GS) {
                int32_t ival_q = 0;
                #pragma omp simd
                for (int k = 0; k < GS; k++) {
                    ival_q += (int32_t)temp_data[j + k] * (int32_t)wq_data[in + j + k];
                }
                ivals_q[j / GS] = ival_q;
            }
            // Gather the scales in a nice consecutive array for SIMD.
            float scales_q[dim / GS];
            #pragma omp simd
            for (int j = 0; j < dim / GS; j++) {
                scales_q[j] = wq_scale[in / GS + j] * temp_scale[j];
            }
            float fvals_q[dim / GS];
            #pragma omp simd
            for (int j = 0; j < dim / GS; j++) {
                fvals_q[j] = ((float) ivals_q[j]);
            }
            float sum_q = 0.0f;
            #pragma omp simd
            for (int j = 0; j < dim / GS; j++) {
                sum_q += fvals_q[j] * scales_q[j];
            }        
            q_data[i] = sum_q;
        }

        float * k_data = data_f32(s->k);
        int8_t * wk_data = data_i8(layer->wk);
        float * wk_scale = layer->wk->scale;
        float * v_data = data_f32(s->v);
        int8_t * wv_data = data_i8(layer->wv);
        float * wv_scale = layer->wv->scale;        
        #pragma omp parallel for private(i)
        for (i = 0; i < kv_dim; i++) {
            int in = i * dim;
    
            int32_t ivals_k[dim / GS];
            int32_t ivals_v[dim / GS];
            for (int j = 0; j < dim; j += GS) {
                int32_t ival_k = 0;
                int32_t ival_v = 0;
                #pragma omp simd
                for (int k = 0; k < GS; k++) {
                    ival_k += (int32_t)temp_data[j + k] * (int32_t)wk_data[in + j + k];
                    ival_v += (int32_t)temp_data[j + k] * (int32_t)wv_data[in + j + k];
                }
                ivals_k[j / GS] = ival_k;
                ivals_v[j / GS] = ival_v;
            }
            float scales_k[dim / GS];
            float scales_v[dim / GS];
            #pragma omp simd
            for (int j = 0; j < dim / GS; j++) {
                scales_k[j] = wk_scale[in / GS + j] * temp_scale[j];
                scales_v[j] = wv_scale[in / GS + j] * temp_scale[j];
            }
            float fvals_k[dim / GS];
            float fvals_v[dim / GS];
            #pragma omp simd
            for (int j = 0; j < dim / GS; j++) {
                fvals_k[j] = ((float) ivals_k[j]);
                fvals_v[j] = ((float) ivals_v[j]);
            }
            float sum_k = 0.0f;
            float sum_v = 0.0f;
            #pragma omp simd
            for (int j = 0; j < dim / GS; j++) {
                sum_k += fvals_k[j] * scales_k[j];
                sum_v += fvals_v[j] * scales_v[j];
            }        
            k_data[i] = sum_k;
            v_data[i] = sum_v;
        }        

        ////

        //////////////////////////////////////
        // RoPE relative positional encoding

        assert(st->huggingface_rope);
        if (st->huggingface_rope) {
            // RoPE relative positional encoding: using complex number rotation like in lm.rs
            for (int i = 0; i < p->n_heads; i++) {
                #pragma omp simd
                for (int j = 0; j < head_size/2; j++) {
                    // Calculate indices for the first and second element in each pair
                    int q_idx1 = (i * head_size) + j;
                    int q_idx2 = (i * head_size) + j + (head_size/2);
                    
                    // For query vector - apply complex number rotation
                    float q0 = q_data[q_idx1];
                    float q1 = q_data[q_idx2];
                    q_data[q_idx1] = q0 * fcr[j] - q1 * fci[j];
                    q_data[q_idx2] = q0 * fci[j] + q1 * fcr[j];
                }
                #pragma omp simd
                for (int j = 0; j < min(head_size/2, kv_dim - i*head_size - head_size/2); j++) {
                    // For key vector - check if this head's key part is within kv_dim
                    int k_idx1 = (i * head_size) + j;
                    int k_idx2 = (i * head_size) + j + (head_size/2);
                    
                    float k0 = k_data[k_idx1];
                    float k1 = k_data[k_idx2];
                    k_data[k_idx1] = k0 * fcr[j] - k1 * fci[j];
                    k_data[k_idx2] = k0 * fci[j] + k1 * fcr[j];
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

                // Note: untested refactor.
                float v0 = q_data[i];
                float v1 = q_data[i+1];
                q_data[i] = v0 * fcr - v1 * fci;
                q_data[i+1] = v0 * fci + v1 * fcr;
                // TODO: split this up just like in the RoPE code above.
                if (i < kv_dim) {
                    float v0 = k_data[i];
                    float v1 = k_data[i+1];
                    k_data[i] = v0 * fcr - v1 * fci;
                    k_data[i+1] = v0 * fci + v1 * fcr;
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
        int h;
        // #pragma omp parallel for private(h) // Disabled due to the use of s->kvtemp.
        for (h = 0; h < p->n_heads; h++) {
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

        float *xb2_data = data_f32(s->xb2);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x_data[i] += xb2_data[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, layer->rms_ffn_weight, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, layer->w1, dim, hidden_dim);
        matmul(s->hb2, s->xb, layer->w3, dim, hidden_dim);

        float *hb_data = data_f32(s->hb);
        float *hb2_data = data_f32(s->hb2);

        // SwiGLU non-linearity
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
        // matmul(s->xb, s->hb, layer->w2, hidden_dim, dim);
        convert_into(temp_q8, s->hb);

        int8_t * w2_data = data_i8(layer->w2);
        float * w2_scale = layer->w2->scale;
    
        #pragma omp parallel for private(i)
        for (i = 0; i < dim; i++) {
            int in = i * hidden_dim;
    
            int32_t ivals_q[hidden_dim / GS];
            for (int j = 0; j < hidden_dim; j += GS) {
                int32_t ival_q = 0;
                #pragma omp simd
                for (int k = 0; k < GS; k++) {
                    ival_q += (int32_t)temp_data[j + k] * (int32_t)w2_data[in + j + k];
                }
                ivals_q[j / GS] = ival_q;
            }
            // Gather the scales in a nice consecutive array for SIMD.
            float scales_q[hidden_dim / GS];
            #pragma omp simd
            for (int j = 0; j < hidden_dim / GS; j++) {
                scales_q[j] = w2_scale[in / GS + j] * temp_scale[j];
            }
            float fvals_q[hidden_dim / GS];
            #pragma omp simd
            for (int j = 0; j < hidden_dim / GS; j++) {
                fvals_q[j] = ((float) ivals_q[j]);
            }
            float sum_q = 0.0f;
            #pragma omp simd
            for (int j = 0; j < hidden_dim / GS; j++) {
                sum_q += fvals_q[j] * scales_q[j];
            }        
            x_data[i] += sum_q;
        }        
    }

    // final rmsnorm
    rmsnorm(x, x, st->rms_final_weight, dim);

    // classifier into logits
    matmul(s->logits, x, st->wcls, p->dim, p->vocab_size);
    return s->logits;
}
