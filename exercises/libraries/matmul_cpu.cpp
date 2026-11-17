#include "matmul.hpp"
#include <omp.h>

/*
 * In column-major storage, elements within a column are contiguous in memory.
 * The loop order is chosen to exploit this:
 *
 *   Outer (j):  select a column of C to fill
 *   Middle (l): step through the sum — at each l, add B(l,j) × column l of A
 *               into column j of C
 *   Inner (i):  walk down the column — A(i,l) and C(i,j) are both
 *               contiguous in memory, so this loop is stride-1 for both
 *
 */

/**
 * @brief Parallel matrix multiplication using OpenMP.
 */
void matmul_cpu(const Matrix& A, const Matrix& B, Matrix& C) {
    int m = A.rows();
    int n = B.cols();
    int k = A.cols();

    C.zero();

    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n; ++j) {
        for (int l = 0; l < k; ++l) {
            for (int i = 0; i < m; ++i) {
                C(i, j) += A(i, l) * B(l, j);
            }
        }
    }
}
