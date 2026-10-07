#include <cstdio>
#include <cstring>
#include <limits>

#include "kernels.hpp"
#include "precision.hpp"
#include "solver.hpp"

#ifdef USE_GPU
#include <cuda_runtime.h>

#include "errors.hpp"
#endif

// Solve A*x = b using the conjugate gradient method.
int cg_solve(real *x, const real *A, const real *b, const int n,
             const int max_iter) {
  real *r, *p, *A_times_p;
#ifdef USE_GPU
  CHECK_CUDA_ERROR(cudaMalloc(&r, n * sizeof(real)));
  CHECK_CUDA_ERROR(cudaMalloc(&p, n * sizeof(real)));
  CHECK_CUDA_ERROR(cudaMalloc(&A_times_p, n * sizeof(real)));
#else
  r = new real[n];
  p = new real[n];
  A_times_p = new real[n];
#endif

  // Step 1: r_0 = b - A*x_0
  matvec(r, A, x, n);
  axpby(r, b, 1.0, -1.0, n); // r = b - r

  // Step 2: p_0 = r_0
#ifdef USE_GPU
  CHECK_CUDA_ERROR(
      cudaMemcpy(p, r, n * sizeof(real), cudaMemcpyDeviceToDevice));
#else
  std::memcpy(p, r, n * sizeof(real));
#endif

  real residual_sq_old = dot(r, r, n);
  int n_iter;

  for (n_iter = 0; n_iter < max_iter; n_iter++) {
    // Step 3a: alpha_k = (r_k . r_k) / (p_k . K*p_k)
    matvec(A_times_p, A, p, n);
    real alpha = residual_sq_old / dot(p, A_times_p, n);

    // Step 3b: x_{k+1} = x_k + alpha_k * p_k
    axpby(x, p, alpha, 1.0, n);

    // Step 3c: r_{k+1} = r_k - alpha_k * K*p_k
    axpby(r, A_times_p, -alpha, 1.0, n);

    real residual_sq_new = dot(r, r, n);

    // This method is so good it crashes if the residual gets too small!
    if (residual_sq_new < std::numeric_limits<real>().epsilon())
      break;

    // Step 3d: beta_k = (r_{k+1} . r_{k+1}) / (r_k . r_k)
    //          p_{k+1} = r_{k+1} + beta_k * p_k
    real beta = residual_sq_new / residual_sq_old;
    axpby(p, r, 1.0, beta, n);

    std::printf("%d: r = %.6e\n", n_iter, residual_sq_new / n);

    residual_sq_old = residual_sq_new;
  }

#ifdef USE_GPU
  CHECK_CUDA_ERROR(cudaFree(r));
  CHECK_CUDA_ERROR(cudaFree(p));
  CHECK_CUDA_ERROR(cudaFree(A_times_p));
#else
  delete[] r;
  delete[] p;
  delete[] A_times_p;
#endif

  return n_iter;
}
