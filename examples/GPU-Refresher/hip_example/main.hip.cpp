#include "hip/hip_runtime.h"
#include <hip/hip_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>

__global__ void vecAdd(float* A, float* B, float* C, int vectorLength) {
    int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if(workIndex < vectorLength) {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}

void initArray(float* A, int length) {
     std::srand(std::time({}));
    for(int i=0; i<length; i++) {
        A[i] = rand() / (float)RAND_MAX;
    }
}

void serialVecAdd(float* A, float* B, float* C,  int length) {
    for(int i=0; i<length; i++) {
        C[i] = A[i] + B[i];
    }
}

bool vectorApproximatelyEqual(float* A, float* B, int length, float epsilon=0.00001) {
    for(int i=0; i<length; i++) {
        if(fabs(A[i] -B[i]) > epsilon) {
            printf("Index %d mismatch: %f != %f", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

//explicit-memory-begin
void explicitMemExample(int vectorLength) {
    // Pointers for host memory
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));
    
    // Pointers for device memory
    float* devA = nullptr;
    float* devB = nullptr;
    float* devC = nullptr;

    //Allocate Host Memory using hipHostMalloc API. This is best practice
    // when buffers will be used for copies between CPU and GPU memory
    hipHostMalloc(&A, vectorLength*sizeof(float));
    hipHostMalloc(&B, vectorLength*sizeof(float));
    hipHostMalloc(&C, vectorLength*sizeof(float));

    // Initialize vectors on the host
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // start-allocate-and-copy
    // Allocate memory on the GPU
    hipMalloc(&devA, vectorLength*sizeof(float));
    hipMalloc(&devB, vectorLength*sizeof(float));
    hipMalloc(&devC, vectorLength*sizeof(float));

    // Copy data to the GPU
    hipMemcpy(devA, A, vectorLength*sizeof(float), hipMemcpyDefault);
    hipMemcpy(devB, B, vectorLength*sizeof(float), hipMemcpyDefault);
    hipMemset(devC, 0, vectorLength*sizeof(float));
    // end-allocate-and-copy

    // Launch the kernel
    int threads = 256;
    int blocks = (vectorLength + threads-1)/threads;
    vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
    // wait for kernel execution to complete
    hipDeviceSynchronize();

    // Copy results back to host
    hipMemcpy(C, devC, vectorLength*sizeof(float), hipMemcpyDefault);

    // Perform computation serially on CPU for comparison
    serialVecAdd(A, B, comparisonResult, vectorLength);

    // Confirm that CPU and GPU got the same answer
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength)) {
        printf("Explicit Memory: CPU and GPU answers match\n");
    } else {
        printf("Explicit Memory: Error - CPU and GPU answers to not match\n");
    }

    // clean up
    hipFree(devA);
    hipFree(devB);
    hipFree(devC);
    hipHostFree(A);
    hipHostFree(B);
    hipHostFree(C);
    free(comparisonResult);
}
//explicit-memory-end

int main(int argc, char** argv) {
    int vectorLength = 1024;
    if(argc >=2) {
        vectorLength = std::atoi(argv[1]);
    }
    explicitMemExample(vectorLength);		
    return 0;
}
