#ifndef BENCHMARKING_H
#define BENCHMARKING_H

#include "tensor.h"

void benchmark_init(int input_size, int output_size, quant_t type);
void benchmark_run(void);
void benchmark_destroy(void);

#endif // BENCHMARKING_H