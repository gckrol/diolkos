/* Inference for Llama-2 Transformer model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/mman.h>
#include <time.h>

#include "transformer.h"
#include "tokenizer.h"
#include "sampler.h"
#include "transformer_info.h"
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include "safetensors.h"
#include "worker_commands.h"
#include <assert.h>
#include "net.h"
#include "fnv1a.h"
#include "utils.h"
#include "tensor.h"
#include "remote_workers.h"

// VS Code shows this as undefined.
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 0
#endif

// Log file for performance metrics
FILE* perf_log = NULL;

static int max(int a, int b) {
    return (a > b) ? a : b;
}

static int num_temp = 3;
static Tensor **temp_q8 = NULL;
static Tensor **temp_f32 = NULL;
void init_temp(int dim, int hidden_dim) {
    temp_q8 = malloc(num_temp * sizeof(Tensor*));
    temp_f32 = malloc(num_temp * sizeof(Tensor*));
    for (int i = 0; i < num_temp; i++) {
        temp_q8[i] = tensor_create(max(dim, hidden_dim), Q8_0);
        temp_f32[i] = tensor_create(max(dim, hidden_dim), F32);
    }
}

static double time_in_ms2(struct timespec *start, struct timespec *end) {
    return (end->tv_sec - start->tv_sec) * 1000.0 +
           (end->tv_nsec - start->tv_nsec) / 1e6;
}

void remote_command(int step_id, Commands command, Tensor** out, int num_out, Tensor** in, int num_in, Tensor** matrix, int num_matrix) {
    // printf("remote_command: %d %d %d %d %d\n", step_id, command, num_out, num_in, num_matrix);
    struct timespec overall_start, send_start, send_end, recv_start, recv_end, overall_end;
    clock_gettime(CLOCK_MONOTONIC, &overall_start);

    for (int i=0;i<num_in;i++) {
        assert(in[i]->type == F32);
    }
    for (int i=0;i<num_matrix;i++) {
        assert(matrix[i]->tensor_id > 0);
        assert(matrix[i]->type == Q8_0);
    }
    for (int i=0;i<num_out;i++) {
        assert(out[i]->type == F32);
    }

    // Initialize arrays to track individual worker times (for debugging)
    double worker_send_times[num_workers];
    double worker_recv_times[num_workers];
    double worker_deq_times[num_workers];

    for (int i=0;i<num_in;i++) {
        convert_into(temp_q8[i], in[i]);
    }

    // Start timing the send phase
    clock_gettime(CLOCK_MONOTONIC, &send_start);
    
    // Send inputs.
    for (int w = 0; w < num_workers; w++) {
        struct timespec worker_send_start, worker_send_end;
        clock_gettime(CLOCK_MONOTONIC, &worker_send_start);
        
        RemoteWorker *worker = &workers[w];
        
        // Prepare all data to be sent in a single writev call
        int num_entries = 1 + num_matrix + num_in*2 + 1;
        struct iovec iov[num_entries];
        uint32_t end_marker = 0xCAFEF00D;
        int c = 0;
        
        uint16_t com = command;
        iov[c].iov_base = &com;
        iov[c++].iov_len = sizeof(com);

        for (int i=0;i<num_matrix;i++) {
            iov[c].iov_base = &matrix[i]->tensor_id;
            iov[c++].iov_len = sizeof(uint32_t);
        }
        
        for (int i=0;i<num_in;i++) {
            size_t data_size = in[i]->dim;
            iov[c].iov_base = temp_q8[i]->data;
            iov[c++].iov_len = data_size;
            
            iov[c].iov_base = temp_q8[i]->scale;
            iov[c++].iov_len = data_size / 32 * sizeof(float);
        }
       
        iov[c].iov_base = &end_marker;
        iov[c++].iov_len = sizeof(end_marker);

        assert(c == num_entries);
        
        // Send all data in one system call
        writev_full(worker->fd, iov, num_entries);
        
        clock_gettime(CLOCK_MONOTONIC, &worker_send_end);
        worker_send_times[w] = time_in_ms2(&worker_send_start, &worker_send_end);
    }
    
    // End timing the send phase
    clock_gettime(CLOCK_MONOTONIC, &send_end);
    double total_send_time = time_in_ms2(&send_start, &send_end);

    // Start timing the receive phase
    clock_gettime(CLOCK_MONOTONIC, &recv_start);

    // Read outputs.
    for (int w = 0; w < num_workers; w++) {
        struct timespec worker_recv_start, worker_recv_end, worker_deq_end;
        clock_gettime(CLOCK_MONOTONIC, &worker_recv_start);
        
        RemoteWorker *worker = &workers[w];

        // Read result data and end marker in a single system call
        int num_entries = num_out*2+1;
        struct iovec iov[num_entries];
        int c = 0;

        Tensor *t[num_out];

        for (int i=0;i<num_out;i++) {
            size_t start = round_down_32(out[i]->dim * worker->start);
            size_t end = round_down_32(out[i]->dim * worker->end);
            size_t length = end - start;

            t[i] = tensor_create(length, Q8_0); // TODO: don't allocate, use temp_q8[i] (if big enough)

            iov[c].iov_base = t[i]->data;
            iov[c++].iov_len = length * sizeof(uint8_t);
    
            iov[c].iov_base = t[i]->scale;
            iov[c++].iov_len = length / 32 * sizeof(float);
        }
            
        uint32_t end_marker = 1;
        iov[c].iov_base = &end_marker;
        iov[c++].iov_len = sizeof(end_marker);

        assert(c == num_entries);
        
        readv_full(worker->fd, iov, num_entries);
        
        // Verify the end marker
        if (end_marker != 0xCAFEF00D) {
            fprintf(stderr, "Error: expected end marker 0xCAFEF00D, got 0x%X\n", end_marker);
            assert(!"missing end marker");
            exit(EXIT_FAILURE);
        }

        clock_gettime(CLOCK_MONOTONIC, &worker_recv_end);
        worker_recv_times[w] = time_in_ms2(&worker_recv_start, &worker_recv_end);

        for (int i=0;i<num_out;i++) {
            size_t start = round_down_32(out[i]->dim * worker->start);
            size_t end = round_down_32(out[i]->dim * worker->end);
            size_t length = end - start;

            convert_q8_f32_slice_into_offset(out[i], t[i], 0, length, start);

            tensor_destroy(t[i]);
        }

        clock_gettime(CLOCK_MONOTONIC, &worker_deq_end);
        worker_deq_times[w] = time_in_ms2(&worker_recv_end, &worker_deq_end);
    }
    
    // End timing the receive phase
    clock_gettime(CLOCK_MONOTONIC, &recv_end);
    double total_recv_time = time_in_ms2(&recv_start, &recv_end);
    
    clock_gettime(CLOCK_MONOTONIC, &overall_end);
    double total_time = time_in_ms2(&overall_start, &overall_end);

    // Calculate GMAC
    double gmac = 0.0;
    for (int i=0;i<num_matrix;i++) {
        gmac += (double)matrix[i]->dim / 1e9 / (total_time / 1000.0);
    }

    // Log performance data if log file is open
    if (perf_log != NULL) {
        fprintf(perf_log, "%d,", step_id);
        for (int w = 0; w < num_workers; w++) {
            fprintf(perf_log, "%.3f,", worker_send_times[w]);
        }
        fprintf(perf_log, "%.3f,", total_send_time);
        for (int w = 0; w < num_workers; w++) {
            fprintf(perf_log, "%.3f,", worker_recv_times[w]);
        }
        for (int w = 0; w < num_workers; w++) {
            fprintf(perf_log, "%.3f,", worker_deq_times[w]);
        }
        fprintf(perf_log, "%.3f,%.3f,%d,%.3f\n", total_recv_time, total_time, 0, gmac);
    }
}

void matmul_remote(Tensor* xoutt, Tensor* xt, Tensor* wt, int in_dim, int out_dim) {
    remote_command(wt->tensor_id, CMD_MULTIPLY, &xoutt, 1, &xt, 1, &wt, 1);
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms(void) {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
// VS Code will complain about CLOCK_REALTIME being undefined otherwise.
#ifndef CLOCK_REALTIME
#define CLOCK_REALTIME 0
#endif
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// Benchmark client overhead using CMD_MULTIPLY_OVERHEAD
float benchmark_overhead(Model *m, RemoteWorker *worker, int iterations, bool perform_matmul) {
    struct timespec starttime, endtime;
    double total_time = 0.0;
    Tensor *matrix = m->layers[0].w1; // Largest matrix we have.
    const int slice_id = matrix->tensor_id;
    
    // Create dummy input data of reasonable size
    const size_t data_size = matrix->hdim;
    uint8_t dummy_data[data_size];
    float dummy_scale[data_size / 32];
    memset(dummy_data, 0, data_size);
    memset(dummy_scale, 0, data_size / 32 * sizeof(float));

    size_t start = round_down_32(matrix->vdim * worker->start);
    size_t end = round_down_32(matrix->vdim * worker->end);

    size_t output_size = end - start;
    output_size += output_size / 32 * sizeof(float); // Scale
    
    for (int i = 0; i < iterations; i++) {
        clock_gettime(CLOCK_MONOTONIC, &starttime);
        
        // Send command using the same pattern as matmul_remote
        struct iovec iov[5];
        uint16_t command = perform_matmul ? CMD_MULTIPLY : CMD_MULTIPLY_OVERHEAD;
        uint32_t end_marker = 0xCAFEF00D;
        
        iov[0].iov_base = &command;
        iov[0].iov_len = sizeof(command);
        
        iov[1].iov_base = (void*)&slice_id;
        iov[1].iov_len = sizeof(slice_id);
        
        iov[2].iov_base = dummy_data;
        iov[2].iov_len = data_size;
        
        iov[3].iov_base = dummy_scale;
        iov[3].iov_len = data_size / 32 * sizeof(float);
        
        iov[4].iov_base = &end_marker;
        iov[4].iov_len = sizeof(end_marker);
        
        // Send all data in one system call
        writev_full(worker->fd, iov, 5);
        
        // Read back results (same pattern as matmul_remote)
        uint8_t dummy_output[output_size];
        end_marker = 1;
        
        struct iovec read_iov[2];
        read_iov[0].iov_base = dummy_output;
        read_iov[0].iov_len = sizeof(dummy_output);
        
        read_iov[1].iov_base = &end_marker;
        read_iov[1].iov_len = sizeof(end_marker);

        // printf("Reading %zu bytes\n", sizeof(dummy_output));
        
        readv_full(worker->fd, read_iov, 2);
        
        // Verify the end marker
        if (end_marker != 0xCAFEF00D) {
            fprintf(stderr, "Error: expected end marker 0xCAFEF00D, got 0x%X\n", end_marker);
            assert(!"missing end marker");
            exit(EXIT_FAILURE);
        }
        
        clock_gettime(CLOCK_MONOTONIC, &endtime);
        double elapsed_ns = (endtime.tv_sec - starttime.tv_sec) * 1e9 + (endtime.tv_nsec - starttime.tv_nsec);
        total_time += elapsed_ns;
    }
    
    double avg_time_ns = total_time / iterations;
    printf("Worker %s:%d: network overhead: %.3f ms (%.3f us) input size: %zu bytes, output size: %zu bytes)\n",
           worker->address, worker->port,
           avg_time_ns / 1e6, avg_time_ns / 1e3, data_size + data_size/32*sizeof(float), output_size);
    return (float)(avg_time_ns / 1e6); // Return value in milliseconds for compatibility
}

// Benchmark network latency using CMD_PING
float benchmark_latency(RemoteWorker *worker, int iterations) {
    struct timespec start, end;
    double total_time = 0.0;
    
    for (int i = 0; i < iterations; i++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        // Send ping command
        uint16_t command = CMD_PING;
        write_full(worker->fd, &command, sizeof(command));
        
        // Wait for response (single byte)
        uint8_t response = 0;
        read_full(worker->fd, &response, sizeof(response));
        
        // Verify response
        if (response != 0x01) {
            fprintf(stderr, "Error: invalid ping response from worker: %d\n", response);
            close(worker->fd);
            exit(EXIT_FAILURE);
        }
        
        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
        total_time += elapsed_ns;
    }
    
    double avg_time_ns = total_time / iterations;
    printf("Worker %s:%d: average round-trip latency: %.3f ms (%.3f us)\n",
           worker->address, worker->port, avg_time_ns / 1e6, avg_time_ns / 1e3);
    return (float)(avg_time_ns / 1e6); // Return value in milliseconds
}

Tensor* forward_remote(Transformer* transformer, int token, int pos) {
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

    // printf("dim: %d, kv_dim: %d, kv_mul: %d, hidden_dim: %d, head_size: %d, attention_dim: %d\n",
    //        dim, kv_dim, kv_mul, hidden_dim, head_size, attention_dim);

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
        // matmul_remote(s->q, s->xb, layer->wq, dim, attention_dim);
        // matmul_remote(s->k, s->xb, layer->wk, dim, kv_dim);
        // matmul_remote(s->v, s->xb, layer->wv, dim, kv_dim);
        Tensor *qkv[3] = {s->q, s->k, s->v};
        Tensor *w[3] = {layer->wq, layer->wk, layer->wv};
        remote_command(layer->wq->tensor_id, CMD_MULTIPLY_QKV, qkv, 3, &s->xb, 1, w, 3);

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
        matmul_remote(s->xb2, s->xb, layer->wo, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            data_f32(x)[i] += data_f32(s->xb2)[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, layer->rms_ffn_weight, dim);

        // FFN non-linearity
        Tensor *matrices[2] = {layer->w1, layer->w3};
        remote_command(layer->w1->tensor_id, CMD_FFN_SILU, &s->hb, 1, &s->xb, 1, matrices, 2);

        matmul_remote(s->xb, s->hb, layer->w2, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            data_f32(x)[i] += data_f32(s->xb)[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, st->rms_final_weight, dim);

    // classifier into logits
    matmul_remote(s->logits, x, st->wcls, p->dim, p->vocab_size);
    return s->logits;
}


// ----------------------------------------------------------------------------
// generation loop

float generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, const char *prompt, int steps) {
    const char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        Tensor* logits = forward_remote(transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, data_f32(logits));
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(tokenizer, token, next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    float tokens_per_second = 0.0f;
    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        long end = time_in_ms();
        int remote_matmuls = transformer->safetensors->config->n_layers * 7;
        tokens_per_second = (float)(pos - 1) * 1000.0f / (end - start);
        fprintf(stderr, "Speed: %.2f tok/s,  %.3f ms/token %.3f ms/rpc\n", tokens_per_second, 1000.0f / tokens_per_second, 1000.0f / tokens_per_second / remote_matmuls);
    }

    free(prompt_tokens);
    return tokens_per_second;
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}

void send_slice(RemoteWorker *worker, int slice_id, Tensor *matrix) {
    if (matrix->type != Q8_0) {
        return; // We run these locally.
    }

    uint32_t start_offset = round_down_32(matrix->vdim * worker->start);
    uint32_t end_offset = round_down_32(matrix->vdim * worker->end);
    // printf("Sending slice %d to worker %s:%d, start: %u, end: %u vdim: %d start: %f end %f\n", slice_id, worker->address, worker->port, start_offset, end_offset, matrix->vdim, worker->start, worker->end);

    // send the slice id and input vector to the worker
    uint16_t command;
    uint32_t type = matrix->type;
    uint32_t dim_in = matrix->hdim;
    uint32_t dim_out = end_offset - start_offset;

    size_t start_index_bytes = start_offset * matrix->hdim * quant_size(matrix->type);
    size_t end_index_bytes = end_offset * matrix->hdim * quant_size(matrix->type);

    size_t bytes_sent = 0;

    u_int8_t *data_start = (uint8_t*)matrix->data + start_index_bytes;
    size_t data_size = end_index_bytes - start_index_bytes;
    assert(data_size % 32 == 0);
    size_t scale_size_bytes = 0;
    if (matrix->type == Q8_0) {
        scale_size_bytes = (end_offset - start_offset) * matrix->hdim / group_size(matrix->type) * sizeof(float);
    }
    // printf("Data size: %d\n", data_size);
    assert(data_start + data_size <= (uint8_t*)matrix->data + Tensor_storage_size(matrix));

    // Check if the worker perhaps has this data already.
    uint128_t hash = fnv1a_128(data_start, data_size);
    if (matrix->type == Q8_0) {
        // We need to hash the scale as well.
        uint8_t *scale_start = (uint8_t*)matrix->scale + start_offset * matrix->hdim / group_size(matrix->type) * sizeof(float);
        assert(scale_start + scale_size_bytes <= (uint8_t*)matrix->scale + Tensor_storage_size(matrix));
        fnv1a_128_continue(scale_start, scale_size_bytes, &hash);
    }
    command = CMD_LOAD_MATRIX_HASH;
    write_full(worker->fd, &command, sizeof(command));
    write_full(worker->fd, &slice_id, sizeof(slice_id));
    write_full(worker->fd, &type, sizeof(type));
    write_full(worker->fd, &dim_in, sizeof(dim_in));
    write_full(worker->fd, &dim_out, sizeof(dim_out));
    write_full(worker->fd, &hash, sizeof(hash));

    uint8_t response;
    read_full(worker->fd, &response, sizeof(response));
    if (response == 0) {
        // The worker has the data already, so we don't need to send it.
        // printf("Worker %d has the data for slice %d\n", worker->fd, slice_id);
        matrix->tensor_id = slice_id;
        return;
    }

    command = CMD_LOAD_MATRIX;
    write_full(worker->fd, &command, sizeof(command));
    write_full(worker->fd, &slice_id, sizeof(slice_id));
    write_full(worker->fd, &type, sizeof(type));
    write_full(worker->fd, &dim_in, sizeof(dim_in));
    write_full(worker->fd, &dim_out, sizeof(dim_out));
    write_full(worker->fd, &hash, sizeof(hash));

    write_full(worker->fd, data_start, data_size);
    bytes_sent += data_size;

    if (matrix->type == Q8_0) {
        float *scale_start = matrix->scale + start_offset * matrix->hdim / group_size(matrix->type);
        size_t scale_size_bytes = (end_offset - start_offset) * matrix->hdim / group_size(matrix->type) * sizeof(float);
        assert((uint8_t*)scale_start + scale_size_bytes <= (float*)matrix->scale + matrix->dim / group_size(matrix->type));

        write_full(worker->fd, scale_start, scale_size_bytes);
        bytes_sent += scale_size_bytes;
    }
    printf("Sent %zu bytes for slice %d Hash: %016lx%016lx\n", bytes_sent, slice_id, hash.high, hash.low);

    write_end_marker(worker->fd);

    matrix->tensor_id = slice_id;
}

void error_usage(void) {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> tokenizer path (default: python/tokenizer.bin)\n");
    fprintf(stderr, "  -m <string> mode: generate|chat (default: generate)\n");
    fprintf(stderr, "  -y <string> system prompt for chat mode (default: NULL)\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin
    const char *tokenizer_path = "python/tokenizer.bin";
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    const char *mode = "generate";    // generate|chat
    char *system_prompt = NULL; // the (optional) system prompt to use in chat mode
    
    // Start timing for initialization
    long start_time = time_in_ms();

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // build the Transformer via the model .bin file
    Transformer transformer = {0};
    build_transformer_from_safetensors(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // override to ~max length
    init_temp(transformer.config.dim, transformer.config.hidden_dim);

    // Print total number of matmul_remote calls for this model
    printf("Total number of matmul_remote calls: %d\n", transformer.config.n_layers * 7);

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // Connect to the workers and upload their matrices.

    // Define the workers. TODO: load from config file, or have them register.
    num_workers = 2;
    workers = calloc(num_workers, sizeof(RemoteWorker));
    workers[0].address = "127.0.0.1";
    workers[0].port = 1234;
    workers[0].start = 0.0f;
    workers[0].end = 0.5f;
    workers[1].address = "127.0.0.1";
    workers[1].port = 1235;
    workers[1].start = 0.5f;
    workers[1].end = 1.0f;

    // // workers[1].address = "192.168.178.12";

    // workers[1].address = "192.168.178.12";
    // // workers[0].address = "192.168.178.36";
    // workers[1].port = 1234;
    // workers[1].start = 0.0f;
    // workers[1].end = 0.4f;

    // workers[0].address = "192.168.178.36";
    // workers[0].port = 1234;
    // workers[0].start = 0.4f;
    // workers[0].end = 1.0f;

    // num_workers = 1;
    // workers = calloc(num_workers, sizeof(RemoteWorker));
    // workers[0].address = "127.0.0.1";
    // workers[0].port = 1234;
    // workers[0].start = 0.0f;
    // workers[0].end = 1.0f;

    // num_workers = 1;
    // workers = calloc(num_workers, sizeof(RemoteWorker));
    // workers[0].address = "192.168.178.36";
    // workers[0].port = 1234;
    // workers[0].start = 0.0f;
    // workers[0].end = 1.0f;

    // Verify start/end of all workers are consecutive
    float total = 0.0f;
    for (int i = 0; i < num_workers; i++) {
        if (workers[i].start < 0.0f || workers[i].end > 1.0f) {
            fprintf(stderr, "Worker %d: start/end must be in [0,1]\n", i);
            exit(EXIT_FAILURE);
        }
        total += workers[i].end - workers[i].start;
        // if (i > 0 && workers[i].start != workers[i - 1].end) {
        //     fprintf(stderr, "Worker %d: start (%f) does not match previous worker end (%f)\n", i, workers[i].start, workers[i - 1].end);
        //     exit(EXIT_FAILURE);
        // }
    }
    if (total < 0.999f || total > 1.001f) {
        fprintf(stderr, "Workers do not cover the full range [0,1]: %f\n", total);
        exit(EXIT_FAILURE);
    }

    // Connect to each of them.
    for (int i = 0; i < num_workers; i++) {
        workers[i].fd = socket(AF_INET, SOCK_STREAM, 0);
        if (workers[i].fd < 0) {
            perror("socket");
            exit(EXIT_FAILURE);
        }

        struct sockaddr_in server_addr;
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(workers[i].port);
        inet_pton(AF_INET, workers[i].address, &server_addr.sin_addr);

        printf("Connecting to worker %d at %s:%d\n", i, workers[i].address, workers[i].port);
        if (connect(workers[i].fd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
            perror("connect");
            exit(EXIT_FAILURE);
        }

        int flag = 1;
        setsockopt(workers[i].fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(int));

        // Busy wait (more CPU, but much lower latency).
        int val = 1;
#ifndef SO_BUSY_POLL
#define SO_BUSY_POLL 1
#endif
        setsockopt(workers[i].fd, SOL_SOCKET, SO_BUSY_POLL, &val, sizeof(val));

        // Loop over all matrices, and send the slice to the worker.
        Model *m = transformer.safetensors;
        int matrix_id = 1;
        send_slice(&workers[i], matrix_id++, m->token_embedding_table);
        send_slice(&workers[i], matrix_id++, m->rms_final_weight);
        send_slice(&workers[i], matrix_id++, m->wcls);
        for (int j = 0; j < m->config->n_layers; j++) {
            Layer *layer = &m->layers[j];
            send_slice(&workers[i], matrix_id++, layer->rms_att_weight);
            send_slice(&workers[i], matrix_id++, layer->rms_ffn_weight);
            send_slice(&workers[i], matrix_id++, layer->wq);
            send_slice(&workers[i], matrix_id++, layer->wk);
            send_slice(&workers[i], matrix_id++, layer->wv);
            send_slice(&workers[i], matrix_id++, layer->wo);
            send_slice(&workers[i], matrix_id++, layer->w1);
            send_slice(&workers[i], matrix_id++, layer->w2);
            send_slice(&workers[i], matrix_id++, layer->w3);
        }
    }

    // Print elapsed initialization time
    long end_time = time_in_ms();
    fprintf(stderr, "Initialization took %ld ms\n", (end_time - start_time));
    
    // Benchmark client overhead for each worker
    for (int i = 0; i < num_workers; i++) {
        benchmark_latency(&workers[i], 1000);
        benchmark_overhead(transformer.safetensors, &workers[i], 1000, false);
        benchmark_overhead(transformer.safetensors, &workers[i], 100, true);
    }

    size_t total_params = print_transformer_info(&transformer);

    //////////////

    // Initialize performance log file
    perf_log = fopen("matmul_perf.csv", "w");
    if (perf_log != NULL) {
        // Write CSV header
        fprintf(perf_log, "tensor_id,");
        for (int i = 0; i < num_workers; i++) {
            fprintf(perf_log, "worker%d_send_time,", i);
        }
        fprintf(perf_log, "total_send_time,");
        for (int i = 0; i < num_workers; i++) {
            fprintf(perf_log, "worker%d_recv_time,", i);
        }
        for (int i = 0; i < num_workers; i++) {
            fprintf(perf_log, "worker%d_deq_time,", i);
        }
        fprintf(perf_log, "total_recv_time,total_time,dim,gmac\n");
    } else {
        fprintf(stderr, "Warning: Failed to open performance log file\n");
    }
        
    //////////////

    float tokens_per_second = generate(&transformer, &tokenizer, &sampler, prompt, steps);
    fprintf(stderr, "GMAC: %.1f\n", (double)tokens_per_second*total_params / 1000000000.0);

    // Close the performance log file
    if (perf_log != NULL) {
        fclose(perf_log);
        fprintf(stderr, "Performance log written to matmul_perf.csv\n");
    }

    // Final benchmark to detect any throttling.
    for (int i = 0; i < num_workers; i++) {
        benchmark_latency(&workers[i], 10);
        benchmark_overhead(transformer.safetensors, &workers[i], 10, false);
        benchmark_overhead(transformer.safetensors, &workers[i], 10, true);
    }

    return 0;
}

