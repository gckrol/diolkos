#include "benchmarking.h"
#include <stdio.h>
#include "tensor.h"
#include <time.h>
#include <stdlib.h> 
#include "threading.h"
#include "utils.h"

// VS Code shows this as undefined.
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 0
#endif

int main(int argc, char *argv[]) {
    struct timespec startup_time;
    clock_gettime(CLOCK_MONOTONIC, &startup_time);
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <size>\nExample: %s 7000\n", argv[0], argv[0]);
        return 1;
    }

    int input_size = round_down_32(atoi(argv[1]));
    int output_size = input_size;
    const int iterations = 400;

    init_threads();
    init_utils(input_size, input_size);

    benchmark_init(input_size, output_size, Q8_0);

    // Warmup.
    for (int i=0;i<iterations/2;i++) {
        benchmark_runp();
    }

    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);
    double startup_elapsed_ms = (start.tv_sec - startup_time.tv_sec) * 1000.0 +
        (start.tv_nsec - startup_time.tv_nsec) / 1e6;
    printf("Startup took %.3f ms\n", startup_elapsed_ms);

    for (;;) {
        clock_gettime(CLOCK_MONOTONIC, &start);  
        for (int i = 0; i < iterations; i++) {
            benchmark_runp();
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
            (end.tv_nsec - start.tv_nsec) / 1e6;
        double gmac = (double)input_size * output_size * iterations / (elapsed_ms / 1000.0) / 1e9;
        printf("Benchmarking took %.3f ms (%.1f GMAC)\n", elapsed_ms, gmac);
    }

    // No need to call benchmark_destroy() here, as the program will exit.
    return 0;
}