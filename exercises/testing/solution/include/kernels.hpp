#pragma once

void run_matvec(float *y, const float *A, const float *x, const int n);
void custom_matvec_wrapper(float *y, const float *A, const float *x,
                           const int n);
void run_get_index(int i, int j, int n, int *result);
