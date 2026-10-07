#include "adders.cuh"
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

// We need a kernel wrapper to fetch the result from the __device__ function
// add_two
__global__ void add_two_wrapper(unsigned int *result, unsigned int number) {
  *result = add_two(number);
}

// Wrap the kernel launch as well to hide data movement boilerplate
unsigned int get_add_two(unsigned int number) {
  unsigned int *result = nullptr;
  cudaMallocManaged(&result, sizeof(*result));
  add_two_wrapper<<<1, 1>>>(result, number);
  cudaDeviceSynchronize(); // make sure kernel finishes
  const unsigned int val =
      *result; // using managed memory so runtime handles movement
  cudaFree(result);
  return val;
}

TEST_CASE("adders works as expected", "[cuda_functions]") {
  REQUIRE(add_one(1) == 2);
  REQUIRE(add_one(100) == 101);

  REQUIRE(get_add_two(1) == 3);
  REQUIRE(get_add_two(100) == 102);
}
