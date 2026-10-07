#include "matmul.hpp"
#include <cblas.h>

/**
 * @brief Matrix multiplication using cblas_dgemm.
 *
 * Matrix data is column-major, matching BLAS's native layout, so no
 * reordering or transpose tricks are needed. The leading dimension of a
 * column-major matrix is its number of rows.
 */
void matmul_blas(const Matrix& A, const Matrix& B, Matrix& C) {
    int m = A.rows();   // rows of A and C
    int n = B.cols();   // cols of B and C
    int k = A.cols();   // cols of A, rows of B

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, k,
                1.0, A.data(), m,
                     B.data(), k,
                0.0, C.data(), m);
}
