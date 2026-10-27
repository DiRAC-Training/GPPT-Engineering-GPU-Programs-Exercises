#include "timer.cuh"
#include <cstdio>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void vecAdd(float *A, float *B, float *C, int vectorLength) {
  int workIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (workIndex < vectorLength) {
    C[workIndex] = A[workIndex] + B[workIndex];
  }
}

void initArray(float *A, int length) {
  std::srand(std::time({}));
  for (int i = 0; i < length; i++) {
    A[i] = rand() / (float)RAND_MAX;
  }
}

int main() {
  const int vectorLength = 1'000'000;

  // Pointers for host memory
  float *A = nullptr;
  float *B = nullptr;
  float *C = nullptr;
  float *comparisonResult = (float *)malloc(vectorLength * sizeof(float));

  // Pointers for device memory
  float *devA = nullptr;
  float *devB = nullptr;
  float *devC = nullptr;

  CUDATimer timer;
  // Allocate Host Memory using cudaHostMalloc API. This is best practice
  //  when buffers will be used for copies between CPU and GPU memory
  cudaMallocHost(&A, vectorLength * sizeof(float));
  cudaMallocHost(&B, vectorLength * sizeof(float));
  cudaMallocHost(&C, vectorLength * sizeof(float));

  // Initialize vectors on the host
  initArray(A, vectorLength);
  initArray(B, vectorLength);

  // Allocate memory on the GPU
  cudaMalloc(&devA, vectorLength * sizeof(float));
  cudaMalloc(&devB, vectorLength * sizeof(float));
  cudaMalloc(&devC, vectorLength * sizeof(float));

  // Copy data to the GPU
  cudaMemcpy(devA, A, vectorLength * sizeof(float), cudaMemcpyDefault);
  cudaMemcpy(devB, B, vectorLength * sizeof(float), cudaMemcpyDefault);
  cudaMemset(devC, 0, vectorLength * sizeof(float));
  // end-allocate-and-copy

  const auto memCopyHostToDeviceTime = timer.lap_ms();
  // Launch the kernel
  int threads = 256;
  int blocks = (vectorLength + threads - 1) / threads;
  vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
  // wait for kernel execution to complete

  const auto kernelTime = timer.lap_ms();

  // Copy results back to host
  cudaMemcpy(C, devC, vectorLength * sizeof(float), cudaMemcpyDefault);

  const auto memCopyDeviceToHostTime = timer.lap_ms();

  // Print timing information
  printf("Host to device memCopy time (ms) %f\n", memCopyHostToDeviceTime);
  printf("Kernel time (ms) %f\n", kernelTime);
  printf("Device to host memCopy time (ms) %f\n", memCopyDeviceToHostTime);
  printf("Total GPU operations time (ms) %f\n",
         memCopyHostToDeviceTime + kernelTime + memCopyDeviceToHostTime);

  // clean up
  cudaFree(devA);
  cudaFree(devB);
  cudaFree(devC);
  cudaFreeHost(A);
  cudaFreeHost(B);
  cudaFreeHost(C);
  free(comparisonResult);
}
