#ifndef UTILS_H
#define UTILS_H

// Common mathematical functions used across different components
void softmax(float* x, int size);
float* rmsnorm(float* o, float* x, float* weight, int size);
void matmul(float* xout, float* x, float* w, int n, int d);

#endif // UTILS_H