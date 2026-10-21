#pragma once

class CUDATimer {
public:
  CUDATimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
  }

  ~CUDATimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(end);
  }

  void reset() { cudaEventRecord(start); }

  float lap_ms() {
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float diff_ms = 0.0f;
    cudaEventElapsedTime(&diff_ms, start, end);
    reset();
    return diff_ms;
  }

private:
  cudaEvent_t start, end;
};
