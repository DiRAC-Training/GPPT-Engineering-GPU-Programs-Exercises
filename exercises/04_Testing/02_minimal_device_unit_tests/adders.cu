#include "adders.cuh"

// This can be tested on the host!
__host__ __device__ unsigned int add_one(unsigned int number) {
  return number + 1;
}

// Let's pretend this cannot be tested with __host__
__device__ unsigned int add_two(unsigned int number) { return number + 2; }
