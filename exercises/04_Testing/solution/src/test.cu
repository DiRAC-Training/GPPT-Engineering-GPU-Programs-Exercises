#include <cstring>
#include <cuda_runtime.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "kernels.cuh"
#include "kernels.hpp"

// Checks that the first `n` elements of `actual` match `expected` to within
// `epsilon`.
void require_output_matches(const float *actual, const float *expected, int n,
                            double epsilon) {
  for (int element = 0; element < n; element++) {
    CAPTURE(element);
    REQUIRE_THAT(actual[element],
                 Catch::Matchers::WithinAbs(expected[element], epsilon));
  }
}

// epsilon is set for the example input. If changing the input you might need to
// set the tolerance differently.
const double epsilon = 1e-12;

// A single source of test data shared by every matvec test below: case 0 is the
// identity matrix (whose product returns the input vector unchanged) and case 1
// is a dense matrix. Keeping the inputs and expected outputs in one place means
// the three testing techniques all check exactly the same thing. GENERATE(0, 1)
// then runs each test once per case and reports them independently.
const int n = 4;

// clang-format off
const float A[2][n * n] = {
  { 1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1 },
  {  1,  2,  3,  4,
     5,  6,  7,  8,
     9, 10, 11, 12,
    13, 14, 15, 16 }
};
// clang-format on
const float x[n] = {0, 1, -2, 3};
const float y_expected[2][n] = {
    {0, 1, -2, 3}, // identity leaves x unchanged
    {8, 16, 24, 32}};
const char *case_name[2] = {"identity", "dense"};

// A forward declaration lets a test launch matvec_kernel directly, even though
// the kernel is defined in another translation unit (kernels.cu). This relies
// on separable compilation, which the build already enables.
__global__ void matvec_kernel(float *y, const float *vals, const float *x,
                              int n);

// TODO: Exercise 1.1: Test matvec_kernel through the provided run_matvec
// wrapper. run_matvec does not manage memory, so allocate GPU-accessible
// (managed) memory and copy the host inputs across.
TEST_CASE("matvec through run_matvec", "[matvec]") {
  const int c = GENERATE(0, 1); // 0 = identity, 1 = dense
  INFO("case: " << case_name[c]);

  // run_matvec does no memory management, so allocate managed memory that both
  // the host and device can see, and copy the host inputs across.
  float *A_man, *x_man, *y_man;
  cudaMallocManaged(&A_man, n * n * sizeof(float));
  cudaMallocManaged(&x_man, n * sizeof(float));
  cudaMallocManaged(&y_man, n * sizeof(float));

  std::memcpy(A_man, A[c], n * n * sizeof(float));
  std::memcpy(x_man, x, n * sizeof(float));

  run_matvec(y_man, A_man, x_man, n);
  cudaDeviceSynchronize();

  require_output_matches(y_man, y_expected[c], n, epsilon);

  cudaFree(A_man);
  cudaFree(x_man);
  cudaFree(y_man);
}

// TODO: Exercise 1.2: Test matvec_kernel through custom_matvec_wrapper. Because
// the wrapper manages device memory itself, the test passes host arrays
// straight in with no CUDA calls of its own.
TEST_CASE("matvec through custom_matvec_wrapper", "[matvec]") {
  const int c = GENERATE(0, 1);
  INFO("case: " << case_name[c]);

  float y[n];
  custom_matvec_wrapper(y, A[c], x, n);

  require_output_matches(y, y_expected[c], n, epsilon);
}

// TODO: Exercise 1.3: Launch matvec_kernel directly. run_matvec fixes the block
// size internally, so launching the kernel yourself is the only way to test it
// under a chosen launch configuration -- here one thread per block, the
// smallest possible, to check the kernel still produces the right answer.
TEST_CASE("matvec through a direct kernel launch", "[matvec]") {
  const int c = GENERATE(0, 1);
  INFO("case: " << case_name[c]);

  float *A_man, *x_man, *y_man;
  cudaMallocManaged(&A_man, n * n * sizeof(float));
  cudaMallocManaged(&x_man, n * sizeof(float));
  cudaMallocManaged(&y_man, n * sizeof(float));

  std::memcpy(A_man, A[c], n * n * sizeof(float));
  std::memcpy(x_man, x, n * sizeof(float));

  const int block_size = 1;
  const int grid = (n + block_size - 1) / block_size;
  matvec_kernel<<<grid, block_size>>>(y_man, A_man, x_man, n);
  cudaDeviceSynchronize();

  require_output_matches(y_man, y_expected[c], n, epsilon);

  cudaFree(A_man);
  cudaFree(x_man);
  cudaFree(y_man);
}

// TODO: Exercise 2.1:
TEST_CASE("get_index from host code", "[get_index]") {

  int result;

  // TODO: Call run_get_index with input parameters & check results
  run_get_index(5, 7, 64, &result);
  REQUIRE(result == 327); // 5 * 64 + 7

  run_get_index(64, 100, 128, &result);
  REQUIRE(result == 8292); // 64 * 128 + 100

  run_get_index(0, 50, 256, &result);
  REQUIRE(result == 50); // 0 * 256 + 50

  run_get_index(37, 0, 1024, &result);
  REQUIRE(result == 37888); // 37 * 1024 + 0
}

// TODO Exercise 2.3: Move your thin wrapper kernel for get_index to this file
__global__ void get_index_wrapper(int i, int j, int n, int *result) {
  *result = get_index(i, j, n);
}

// TODO: Exercise 2.2: Test your get_index wrapper function
TEST_CASE("get_index through a device wrapper", "[get_index]") {

  // TODO: Declare result and make it accessible to the GPU
  int *result;
  cudaMallocManaged(&result, sizeof(int));

  // TODO: Call get_index_wrapper with input parameters & check results
  get_index_wrapper<<<1, 1>>>(5, 7, 64, result);
  cudaDeviceSynchronize();
  REQUIRE(*result == 327); // 5 * 64 + 7

  get_index_wrapper<<<1, 1>>>(64, 100, 128, result);
  cudaDeviceSynchronize();
  REQUIRE(*result == 8292); // 64 * 128 + 100

  get_index_wrapper<<<1, 1>>>(0, 50, 256, result);
  cudaDeviceSynchronize();
  REQUIRE(*result == 50); // 0 * 256 + 50

  get_index_wrapper<<<1, 1>>>(37, 0, 1024, result);
  cudaDeviceSynchronize();
  REQUIRE(*result == 37888); // 37 * 1024 + 0

  cudaFree(result);
}
