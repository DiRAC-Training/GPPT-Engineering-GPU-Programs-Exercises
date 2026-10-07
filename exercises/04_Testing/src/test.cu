#include <cstring>
#include <cuda_runtime.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

// TODO: Exercise 2.3: Uncomment to include the device function declaration
// #include "kernels.cuh"

#include "kernels.hpp"

// Provided helper: checks that the first `n` elements of `actual` match
// `expected` to within `epsilon`.
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

// Provided test data, shared by every matvec test: case 0 is the identity
// matrix (whose product returns the input vector unchanged) and case 1 is a
// dense matrix. GENERATE(0, 1) runs each test once per case and reports the two
// cases independently, so a failure in one still shows the result of the other.
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

// TODO: Exercise 1.1: Test matvec_kernel through the provided run_matvec
// wrapper. run_matvec does not manage memory, so allocate GPU-accessible
// (managed) memory and copy the host inputs across. See src/main.cpp for an
// example using cudaMallocManaged.
TEST_CASE("matvec through run_matvec", "[matvec]") {
  SKIP("Remove this line after completing the exercise TODOs");

  const int c = GENERATE(0, 1); // 0 = identity, 1 = dense
  INFO("case: " << case_name[c]);

  // TODO: Allocate managed memory and copy A[c] and x into it

  // TODO: Call run_matvec with the managed inputs

  // TODO: Check the result against y_expected[c] (e.g. require_output_matches)

  // TODO: Free the managed memory
}

// TODO: Exercise 1.2: Test matvec_kernel through custom_matvec_wrapper (which
// you implement in kernels.cu). Because the wrapper manages device memory
// itself, the test passes host arrays straight in with no CUDA calls of its own.
TEST_CASE("matvec through custom_matvec_wrapper", "[matvec]") {
  SKIP("Remove this line after completing the exercise TODOs");

  const int c = GENERATE(0, 1);
  INFO("case: " << case_name[c]);

  // TODO: Declare a host output array and call custom_matvec_wrapper with A[c]
  // and x

  // TODO: Check the result against y_expected[c]
}

// A forward declaration lets this test launch matvec_kernel directly, even
// though the kernel is defined in another translation unit (kernels.cu). This
// relies on separable compilation, which the build already enables.
__global__ void matvec_kernel(float *y, const float *vals, const float *x,
                              int n);

// TODO: Exercise 1.3: Launch matvec_kernel directly. run_matvec fixes the block
// size internally; launching the kernel yourself lets you choose the launch
// configuration -- try one thread per block, the smallest possible, and check
// the kernel still produces the right answer.
TEST_CASE("matvec through a direct kernel launch", "[matvec]") {
  SKIP("Remove this line after completing the exercise TODOs");

  const int c = GENERATE(0, 1);
  INFO("case: " << case_name[c]);

  // TODO: Allocate managed memory and copy A[c] and x into it

  // TODO: Choose a block size (e.g. 1) and grid size, then launch
  //       matvec_kernel<<<grid, block>>> and synchronise

  // TODO: Check the result against y_expected[c]

  // TODO: Free the managed memory
}

// TODO: Exercise 2.1:
TEST_CASE("get_index from host code", "[get_index]") {
  SKIP("Remove this line after completing the exercise TODOs");

  // TODO: Declare result

  // TODO: Call run_get_index with input parameters & check results
}

// TODO: Exercise 2.2: Forward declare your wrapper function

// TODO: Exercise 2.3: Move your thin wrapper kernel for get_index to this file

// TODO: Exercise 2.2: Test your get_index wrapper function
TEST_CASE("get_index through a device wrapper", "[get_index]") {
  SKIP("Remove this line after completing the exercise TODOs");

  // TODO: Declare result and make it accessible to the GPU

  // TODO: Call get_index_wrapper with input parameters & check results
}
