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

static Tensor *temp_q8 = NULL;

static int max(int a, int b) {
    return (a > b) ? a : b;
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

    // copy the token embedding into x
    convert_slice_into(x, st->token_embedding_table, token * dim, dim);

    // forward all the layers
    for(int l = 0; l < p->n_layers; l++) {
        // access the layer weights
        Layer *layer = &st->layers[l];

        //////////////////////
        // attention rmsnorm
        // rmsnorm(s->xb, x, layer->rms_att_weight, dim);

        float *restrict xb_data = data_f32(s->xb);
        float *restrict x_data = data_f32(x);
        float *restrict weight_data = data_f32(layer->rms_att_weight);
    
        // calculate sum of squares
        float ss = 0.0f;
        for (int j = 0; j < dim; j++) {
            ss += x_data[j] * x_data[j];
        }
        ss /= dim;
        ss += 1e-5f;
        ss = 1.0f / sqrtf(ss);
        // normalize and scale
        for (int j = 0; j < dim; j++) {
            xb_data[j] = weight_data[j] * (ss * x_data[j]);
        }        

        //////////////////////////////////
        // qkv matmuls for this position

        // key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience

        // F32, F32, Q8_0

        // Quantize xb into temp_q8 so we can do a q8 matrix multiplication.
        // convert_into(temp_q8, s->xb);
        // void convert_f32_q8_slice_into_offset(Tensor *dst, Tensor *input, size_t start, size_t length, size_t dst_offset);
    
        int8_t *output_data = data_i8(temp_q8);
    
        size_t it;
        #pragma omp parallel for private(it)
        for (it = 0; it < s->xb->dim; it += GS) {
            float max_val = 0.0f;
            #pragma omp simd
            for (int j = 0; j < GS; j++) {
                max_val = fmaxf(max_val, fabsf(xb_data[it + j]));
            }        
            max_val /= Q_MAXF;
            temp_q8->scale[it / GS] = max_val;
    
            #pragma omp simd
            for (int j = 0; j < GS; j++) {
                output_data[it + j] = (int8_t) roundf(xb_data[it + j] / max_val);
            }
        }

        ////

        // matmul(s->q, s->xb, layer->wq, dim, attention_dim);
        // void matmul_Q8_0(Tensor* xoutt, Tensor* xt, Tensor* wt, int n, int d)

        float *restrict q_data = data_f32(s->q);
        int8_t *restrict xb_q_data = data_i8(temp_q8);
        float *restrict xb_q_scale = temp_q8->scale;
        int8_t *restrict wq_data = data_i8(layer->wq);
        float *restrict wq_scale = layer->wq->scale;
    
        int i;
        #pragma omp parallel for private(i)
        for (i = 0; i < attention_dim; i++) {
            int in = i * dim;
    
            // Implemented as a bunch of small groups, so the compiler will
            // have an easier time vectorizing them.
    
            int32_t ivals[dim / GS];
            for (int j = 0; j < dim; j += GS) {
                int32_t ival = 0;
                #pragma omp simd
                for (int k = 0; k < GS; k++) {
                    ival += (int32_t)xb_q_data[j + k] * (int32_t)wq_data[in + j + k];
                }
                ivals[j / GS] = ival;
            }
            // Gather the scales in a nice consecutive array for SIMD.
            float scales[dim / GS];
            #pragma omp simd
            for (int j = 0; j < dim; j += GS) {
                scales[j / GS] = wq_scale[(in + j) / GS] * xb_q_scale[j / GS];
            }
            // for (int j = 0; j < dim / GS; j++) {
            //     scales[j] = wq_scale[(in + j * GS) / GS] * xb_q_scale[j];
            // }
            float fvals[dim / GS];
            #pragma omp simd
            for (int j = 0; j < dim / GS; j++) {
                fvals[j] = ((float) ivals[j]);
            }
            float sum = 0.0f;
            #pragma omp simd
            for (int j = 0; j < dim / GS; j++) {
                sum += fvals[j] * scales[j];
            }        
            q_data[i] = sum;
        }        

        ////
        matmul(s->k, s->xb, layer->wk, dim, kv_dim);
        matmul(s->v, s->xb, layer->wv, dim, kv_dim);

        //////////////////////////////////////
        // RoPE relative positional encoding

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
                convert_slice_into(s->kvtemp, s->key_cache, loff + t * kv_dim + (h / kv_mul) * head_size, head_size);
                float *k = data_f32(s->kvtemp);

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
            softmax_f32(att, pos + 1);

            // weighted sum of the values, store back into xb
            float* xb = data_f32(s->xb) + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                convert_slice_into(s->kvtemp, s->value_cache, loff + t * kv_dim + (h / kv_mul) * head_size, head_size);
                float *v = data_f32(s->kvtemp);

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
            data_f32(x)[i] += data_f32(s->xb2)[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, layer->rms_ffn_weight, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, layer->w1, dim, hidden_dim);
        matmul(s->hb2, s->xb, layer->w3, dim, hidden_dim);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = data_f32(s->hb)[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= data_f32(s->hb2)[i];
            data_f32(s->hb)[i] = val;
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
