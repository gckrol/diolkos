#define _GNU_SOURCE     // Must be defined before any includes for GNU extensions
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sched.h>      // For CPU_SET, CPU_ZERO, cpu_set_t
#include <unistd.h>     // For sysconf
#include <sched.h>
#include <string.h>

#include "tensor.h"
#include "utils.h"

int num_threads;

typedef struct {
    Tensor* out_tensor;
    Tensor* in_tensor;
    Tensor* matrix;
    Tensor* temp_q8;
    Tensor* temp_f32;
    int start_row;
    int end_row;
    int tid;
    bool quantize;
} ThreadContext;

ThreadContext *contexts = NULL;
pthread_t *threads = NULL;

pthread_barrier_t start_barrier;
pthread_barrier_t end_barrier;

void matmul_Q8_0_slice(Tensor* out_tensor, Tensor* in_tensor, Tensor* matrix, int in_dim, int out_dim, int start_row, int end_row, int offset) {
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
        xout_data[i-start_row+offset] = sum;
    }
}

void *worker_fn(void *arg) {
    ThreadContext *ctx = (ThreadContext *)arg;
    
    // Set CPU affinity for this thread
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(ctx->tid, &cpuset);

    struct sched_param param = { .sched_priority = 50 };
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &param) != 0) {
        perror("pthread_setschedparam");
        fprintf(stderr, "For optimal performance give CAP_SYS_NICE with: sudo setcap cap_sys_nice=eip <path to binary>\n");
    }
    
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        fprintf(stderr, "Warning: Failed to set thread affinity for thread %d\n", ctx->tid);
    }

    while (1) {
        // Wait at barrier until work is available
        pthread_barrier_wait(&start_barrier);

        if (ctx->quantize) {
            matmul_Q8_0_slice(ctx->temp_f32, ctx->in_tensor, ctx->matrix, 
                ctx->in_tensor->dim, ctx->out_tensor->dim, 
                ctx->start_row, ctx->end_row, 0);

            convert_f32_q8_slice_into_offset(ctx->out_tensor, ctx->temp_f32, 
                0, ctx->end_row - ctx->start_row, 
                ctx->start_row);
        } else {
            // Directly into the output tensor.
            matmul_Q8_0_slice(ctx->out_tensor, ctx->in_tensor, ctx->matrix, 
                ctx->in_tensor->dim, ctx->out_tensor->dim, 
                ctx->start_row, ctx->end_row, ctx->start_row);
        }
        
        // Wait at barrier until all threads complete
        pthread_barrier_wait(&end_barrier);
    }

    return NULL;
}

void matmul_parallel(Tensor* out_tensor, Tensor* in_tensor, Tensor* matrix) {
    // Set up work for all threads
    for (int i = 0; i < num_threads; ++i) {
        contexts[i].in_tensor = in_tensor;
        contexts[i].out_tensor = out_tensor;
        contexts[i].matrix = matrix;
        float start = (float)i / num_threads;
        float end = (float)(i + 1) / num_threads;
        contexts[i].start_row = round_down_32(start * out_tensor->dim);
        contexts[i].end_row = round_down_32(end * out_tensor->dim);
        contexts[i].quantize = true;
    }
    
    // Trigger all threads to start working
    pthread_barrier_wait(&start_barrier);
    
    // Wait for all threads to complete
    pthread_barrier_wait(&end_barrier);
}

void matmul_parallel_f32(Tensor* out_tensor, Tensor* in_tensor, Tensor* matrix) {
    // Set up work for all threads
    for (int i = 0; i < num_threads; ++i) {
        contexts[i].in_tensor = in_tensor;
        contexts[i].out_tensor = out_tensor;
        contexts[i].matrix = matrix;
        float start = (float)i / num_threads;
        float end = (float)(i + 1) / num_threads;
        contexts[i].start_row = round_down_32(start * out_tensor->dim);
        contexts[i].end_row = round_down_32(end * out_tensor->dim);
        contexts[i].quantize = false;
    }
    
    // Trigger all threads to start working
    pthread_barrier_wait(&start_barrier);
    
    // Wait for all threads to complete
    pthread_barrier_wait(&end_barrier);
}

void init_threads() {
    // Read OMP_NUM_THREADS from environment
    char *env = getenv("OMP_NUM_THREADS");
    num_threads = sysconf(_SC_NPROCESSORS_ONLN); // Default to number of processors
    if (env && strlen(env) > 0) {
        int n = atoi(env);
        if (n > 0) num_threads = n;
    }
    printf("Using %d threads\n", num_threads);

    // Allocate arrays based on num_threads
    contexts = calloc(num_threads, sizeof(ThreadContext));
    threads = calloc(num_threads, sizeof(pthread_t));

    // Initialize barriers
    pthread_barrier_init(&start_barrier, NULL, num_threads + 1); // +1 for main thread
    pthread_barrier_init(&end_barrier, NULL, num_threads + 1);   // +1 for main thread
    
    for (int i = 0; i < num_threads; ++i) {
        contexts[i].temp_q8 = tensor_create(1024*100, Q8_0); // FIXME SIZE.
        contexts[i].temp_f32 = tensor_create(1024*100, F32); // FIXME SIZE.
        contexts[i].tid = i;
        
        pthread_create(&threads[i], NULL, worker_fn, &contexts[i]);
    }
}
