#include <cstdio>
#include <cstdlib>
#include <chrono>

// Number of elements (64 million floats = 256 MB)
const int N = 1 << 26;

// Each GPU thread adds 1 to one element
__global__ void addOne(float* data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = data[i] + 1.0f;
    }
}

// Check that every element equals 2
bool check(float* data, int n)
{
    for (int i = 0; i < n; i++) {
        if (data[i] != 2.0f) {
            return false;
        }
    }
    return true;
}

// Version 1: manual memory transfers
double runManual()
{
    auto start = std::chrono::high_resolution_clock::now();

    // Allocate on the CPU and on the GPU
    float* hostData = (float*)malloc(N * sizeof(float));
    float* deviceData;
    cudaMalloc(&deviceData, N * sizeof(float));

    // Fill on the CPU
    for (int i = 0; i < N; i++) {
        hostData[i] = 1.0f;
    }

    // Copy CPU -> GPU
    cudaMemcpy(deviceData, hostData, N * sizeof(float), cudaMemcpyHostToDevice);

    // Run the kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    addOne<<<blocks, threads>>>(deviceData, N);

    // Copy GPU -> CPU (this also waits for the kernel)
    cudaMemcpy(hostData, deviceData, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Check on the CPU
    bool ok = check(hostData, N);

    // Free memory
    cudaFree(deviceData);
    free(hostData);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    printf("Manual:  %s\n", ok ? "correct" : "WRONG");
    return elapsed.count();
}

// Version 2: managed (unified) memory
double runManaged()
{
    auto start = std::chrono::high_resolution_clock::now();

    // One allocation, usable from both CPU and GPU
    float* data;
    cudaMallocManaged(&data, N * sizeof(float));

    // Fill on the CPU
    for (int i = 0; i < N; i++) {
        data[i] = 1.0f;
    }

    // Run the kernel (data moves to the GPU automatically)
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    addOne<<<blocks, threads>>>(data, N);

    // Wait for the GPU to finish before the CPU reads
    cudaDeviceSynchronize();

    // Check on the CPU (data moves back automatically)
    bool ok = check(data, N);

    // Free memory
    cudaFree(data);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    printf("Managed: %s\n", ok ? "correct" : "WRONG");
    return elapsed.count();
}

int main()
{
    // Wake up the GPU so start-up cost is not timed
    cudaFree(0);

    double manualTime = runManual();
    double managedTime = runManaged();

    printf("\n");
    printf("Manual transfers: %8.2f ms\n", manualTime);
    printf("Managed memory:   %8.2f ms\n", managedTime);

    // Catch any CUDA error
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
