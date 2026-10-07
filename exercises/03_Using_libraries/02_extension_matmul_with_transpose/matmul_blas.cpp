#include "matmul.hpp"

// TODO: Add necessary includes

/**
 * @brief Matrix multiplication C = A * B using a BLAS library.
 *
 * Consult the BLAS documentation for dgemm.
 * Consider how the library expects matrix data to be laid out.
 */
void matmul_blas(const Matrix& A, const Matrix& B, Matrix& C) {
    int m = A.rows();   // rows of A and C
    int n = B.cols();   // cols of B and C
    int k = A.cols();   // cols of A, rows of B

    // TODO: Implement using BLAS
    (void)m; (void)n; (void)k;  // suppress unused warnings
}
