#define _GNU_SOURCE     // Must be defined before any includes for GNU extensions
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sched.h>      // For CPU_SET, CPU_ZERO, cpu_set_t
#include <unistd.h>     // For sysconf

#include "tensor.h"

#define NUM_THREADS 4

typedef struct {
    Tensor* out_tensor;
    Tensor* in_tensor;
    Tensor* matrix;
    Tensor* temp_q8;
    int start_row;
    int end_row;
    int tid;
} ThreadContext;

pthread_barrier_t start_barrier;
pthread_barrier_t end_barrier;

ThreadContext contexts[NUM_THREADS];
pthread_t threads[NUM_THREADS];

void matmul_Q8_0_slice(Tensor* out_tensor, Tensor* in_tensor, Tensor* matrix, int in_dim, int out_dim, int start_row, int end_row) {
    // printf("matmul_Q8_0_slice: in: %zu xout: %zu in_dim: %d out_dim: %d start_row: %d end_row: %d\n", in_tensor->dim, out_tensor->dim, in_dim, out_dim, start_row, end_row);
    float * xout_data = __builtin_assume_aligned(data_f32(out_tensor), 32);
    int8_t * x_data = __builtin_assume_aligned(data_i8(in_tensor), 32);
    float * x_scale = __builtin_assume_aligned(in_tensor->scale, 32);
    int8_t * w_data = __builtin_assume_aligned(data_i8(matrix), 32);
    float * w_scale = __builtin_assume_aligned(matrix->scale, 32);

    const int GS = 32;

    for (int i = start_row; i < end_row; i++) {
        int in = i * in_dim;

        float sum = 0.0f;
        for (int j = 0; j < in_dim / GS; j++) {
            int8_t *x_start = __builtin_assume_aligned(x_data + j * GS, 32);
            int8_t *w_start = __builtin_assume_aligned(w_data + in + j * GS, 32);

            int32_t ival = 0;
            #pragma omp simd simdlen(16)
            for (int k = 0; k < GS; k++) {
                ival += (int32_t)x_start[k] * (int32_t)w_start[k];
            }
            // It's faster to do this right away, in this loop.
            // This might be because we can do the multiplication while waiting for memory.
            sum += ((float) ival) * w_scale[in / GS + j] * x_scale[j];
        }        
        xout_data[i] = sum;
    }
}

void *worker_fn(void *arg) {
    ThreadContext *ctx = (ThreadContext *)arg;
    
    // Set CPU affinity for this thread
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(ctx->tid, &cpuset);
    
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        fprintf(stderr, "Warning: Failed to set thread affinity for thread %d\n", ctx->tid);
    }

    while (1) {
        // Wait at barrier until work is available
        pthread_barrier_wait(&start_barrier);
        
        // Do assigned chunk
        convert_into(ctx->temp_q8, ctx->in_tensor);
        matmul_Q8_0_slice(ctx->out_tensor, ctx->temp_q8, ctx->matrix, 
                         ctx->in_tensor->dim, ctx->out_tensor->dim, 
                         ctx->start_row, ctx->end_row);
        
        // Wait at barrier until all threads complete
        pthread_barrier_wait(&end_barrier);
    }

    return NULL;
}

void matmul_parallel(Tensor* out_tensor, Tensor* in_tensor, Tensor* matrix, int in_dim, int out_dim) {
    // Set up work for all threads
    for (int i = 0; i < NUM_THREADS; ++i) {
        contexts[i].in_tensor = in_tensor;
        contexts[i].out_tensor = out_tensor;
        contexts[i].matrix = matrix;
        float start = (float)i / NUM_THREADS;
        float end = (float)(i + 1) / NUM_THREADS;
        contexts[i].start_row = start * out_dim;
        contexts[i].end_row = end * out_dim;
    }
    
    // Trigger all threads to start working
    pthread_barrier_wait(&start_barrier);
    
    // Wait for all threads to complete
    pthread_barrier_wait(&end_barrier);
}

void init_threads() {
    // Initialize barriers
    pthread_barrier_init(&start_barrier, NULL, NUM_THREADS + 1); // +1 for main thread
    pthread_barrier_init(&end_barrier, NULL, NUM_THREADS + 1);   // +1 for main thread
    
    for (int i = 0; i < NUM_THREADS; ++i) {
        contexts[i].temp_q8 = tensor_create(1024*100, Q8_0); // FIXME SIZE.
        contexts[i].tid = i;
        
        pthread_create(&threads[i], NULL, worker_fn, &contexts[i]);
    }
}
