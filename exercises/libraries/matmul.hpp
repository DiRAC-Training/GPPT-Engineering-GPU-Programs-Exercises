#ifndef MATMUL_HPP
#define MATMUL_HPP

#include "matrix.hpp"

void matmul_cpu(const Matrix& A, const Matrix& B, Matrix& C);
void matmul_blas(const Matrix& A, const Matrix& B, Matrix& C);
void matmul_cublas(const Matrix& A, const Matrix& B, Matrix& C);
void matmul_rocblas(const Matrix& A, const Matrix& B, Matrix& C);

#endif // MATMUL_HPP
