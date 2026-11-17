#include "adders.cuh"
#include <cassert>
#include <cstdio>
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

int main() {
  assert(add_one(1) == 2);
  assert(add_one(100) == 101);

  assert(get_add_two(1) == 3);
  assert(get_add_two(100) == 102);

  printf("All tests passed!\n");

  return 0;
}
