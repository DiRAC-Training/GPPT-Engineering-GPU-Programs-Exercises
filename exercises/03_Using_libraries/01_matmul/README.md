# Exercise: Porting GEMM to BLAS and cuBLAS

## Introduction

This exercise ports a naive matrix multiplication (GEMM) to library implementations: first to CBLAS on the CPU, then to cuBLAS on the GPU.

The operation computes $C = A \times B$, or element-wise:

$$C_{ij} = \sum_{l=0}^{K-1} A_{il} \cdot B_{lj}, \qquad i = 0,\ldots,M-1,\ \ j = 0,\ldots,N-1$$

where $A$ is $M \times K$, $B$ is $K \times N$, and $C$ is $M \times N$.

A working CPU implementation using OpenMP is provided in `matmul_cpu.cpp` as a reference and baseline for verification. Your task is to implement the same operation using CBLAS and cuBLAS.

## Structure

```bash
matrix.hpp           # Matrix class with contiguous column-major storage
matmul.hpp           # Function declarations
matmul_cpu.cpp       # CPU implementation with parallel threading using OpenMP (provided)
matmul_blas.cpp      # BLAS implementation (your task)
matmul_cublas.cu     # cuBLAS implementation (your task)
solve.cpp            # Main driver (provided)
solution/            # Reference solutions
```

## Part 1: CBLAS

The first part of this exercise is to refactor the code to use the BLAS library. Start by opening `matmul_blas.cpp`. The function signature and dimension variables are already set up — your task is to replace the `TODO` with a call to the appropriate CBLAS GEMM function.  

**Hint:** If you are stuck at any step, you can inspect the solution file [solution/matmul_blas.cpp](solution/matmul_blas.cpp) to find the answer.

### Step 1: Identify the correct function

BLAS GEMM functions exist for several numeric types, each denoted by a single-character prefix:

| Prefix | Type                     |
|--------|--------------------------|
| `s`    | single precision float   |
| `d`    | double precision float   |
| `c`    | single precision complex |
| `z`    | double precision complex |

The prefix is appended to form the function name, e.g. `cblas_sgemm` for single precision, `cblas_zgemm` for double precision complex.

**Task:** Check the numeric type used in the `Matrix` class (defined in `matrix.hpp`) to determine which version of the GEMM function to call.

### Step 2: Understand the function signature

The CBLAS GEMM function computes $C = \alpha \cdot \text{op}(A) \times \text{op}(B) + \beta \cdot C$, where $\text{op}(X)$ can be $X$ or $X^T$ (transpose). The full signature is:

```c
cblas_dgemm(layout, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
```

The parameters are:

| Parameter | Description                                                        |
|-----------|--------------------------------------------------------------------|
| `layout`  | Memory layout of the matrices (`CblasRowMajor` or `CblasColMajor`) |
| `transA`  | Whether to transpose A (`CblasNoTrans` or `CblasTrans`)            |
| `transB`  | Whether to transpose B (`CblasNoTrans` or `CblasTrans`)            |
| `m`       | Rows of op(A) and C                                                |
| `n`       | Columns of op(B) and C                                             |
| `k`       | Columns of op(A), rows of op(B)                                    |
| `alpha`   | Scalar multiplier for the A*B product                              |
| `A`       | Pointer to matrix A data                                           |
| `lda`     | Leading dimension of A                                             |
| `B`       | Pointer to matrix B data                                           |
| `ldb`     | Leading dimension of B                                             |
| `beta`    | Scalar multiplier for C                                            |
| `C`       | Pointer to matrix C data                                           |
| `ldc`     | Leading dimension of C                                             |

Here, op(X) denotes matrix X after applying the transpose operation specified by the corresponding `trans` parameter — either the original matrix or its transpose.

Work through the following to determine the correct values:

**Layout:** The `Matrix` class stores data in **column-major** order (look at the indexing in `operator()` to confirm this). Which layout value should you use?

**Transpose:** We want to compute $C = A \times B$ *without* transposing. What transpose values should you pass for A and B?

**Scalars:** The general operation is $C = \alpha \cdot A \times B + \beta \cdot C$. We want a straightforward $C = A \times B$. What values of $\alpha$ and $\beta$ achieve this?

**Leading dimensions:** The "leading dimension" is the stride between consecutive columns in memory. For a column-major matrix, this is the number of rows. Look at the dimension variables already set up in `matmul_blas.cpp`:

```c
int m = A.rows();   // rows of A and C
int n = B.cols();   // cols of B and C
int k = A.cols();   // cols of A, rows of B
```

What are the leading dimensions of A, B and C?

**Data pointers:** Use `A.data()`, `B.data()` and `C.data()` to pass pointers to the underlying contiguous storage.

For full documentation, see:

- [Netlib CBLAS cblas_dgemm](https://www.netlib.org/lapack/explore-html/dc/d18/cblas__dgemm_8c.html)
- [Intel oneMKL cblas_dgemm](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/cblas-gemm-001.html)

### Step 3: Implement and test

**Task:** Implement the CBLAS GEMM call in `matmul_blas.cpp` using the values you determined in Step 2. Uncomment the BLAS section in `solve.cpp` to enable testing.

Build and run:

```bash
cmake -B build
cmake --build build
./build/bin/solve         # Small test (512 x 512)
```

The output should show `Verification: PASSED`.

## Part 2: cuBLAS

The second part of this exercise is to implement the same GEMM operation using cuBLAS on the GPU. Start by opening `matmul_cublas.cu`. The function signature and dimension variables are already set up, as in Part 1, but the GPU version requires additional steps to manage device memory and a library handle.

**Hint:** If you are stuck at any step, you can inspect the solution file [solution/matmul_cublas.cu](solution/matmul_cublas.cu) to find the answer.

### Step 1: Identify the correct function

The cuBLAS GEMM functions follow the same type-prefix naming convention as CBLAS (see Part 1, Step 1). The cuBLAS equivalent uses the same prefix you identified earlier.

**Task:** Find the correct cuBLAS GEMM function name. The cuBLAS documentation is here:

- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/index.html)
- [cublasDgemm reference](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm)

### Step 2: Understand the function signature

The cuBLAS GEMM signature is similar to CBLAS but has some key differences:

```c
cublasDgemm(handle, transA, transB, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
```

The parameters are:

| Parameter | Description                                             |
|-----------|---------------------------------------------------------|
| `handle`  | cuBLAS library handle (see Step 3)                      |
| `transA`  | Whether to transpose A (`CUBLAS_OP_N` or `CUBLAS_OP_T`) |
| `transB`  | Whether to transpose B (`CUBLAS_OP_N` or `CUBLAS_OP_T`) |
| `m`       | Rows of op(A) and C                                     |
| `n`       | Columns of op(B) and C                                  |
| `k`       | Columns of op(A), rows of op(B)                         |
| `&alpha`  | **Pointer** to scalar multiplier for the A*B product    |
| `A`       | Pointer to matrix A data **on the device**              |
| `lda`     | Leading dimension of A                                  |
| `B`       | Pointer to matrix B data **on the device**              |
| `ldb`     | Leading dimension of B                                  |
| `&beta`   | **Pointer** to scalar multiplier for C                  |
| `C`       | Pointer to matrix C data **on the device**              |
| `ldc`     | Leading dimension of C                                  |

Note two differences from CBLAS:

- **No layout parameter.** cuBLAS always assumes column-major layout.
- **Scalars are passed by pointer** (`&alpha`, `&beta`), not by value. Note that in C, the `&` operator takes the address of a variable and returns a pointer, unlike in C++ where `&` in a function signature denotes a reference.

The transpose values, dimensions and leading dimensions are the same as in Part 1. The matrix pointers, however, must point to **device memory** — see Step 4.

### Step 3: Create and destroy a cuBLAS handle

cuBLAS requires a handle that manages the library's internal state. Create it before any cuBLAS calls and destroy it when you are done:

```c
cublasHandle_t handle;
cublasCreate(&handle);

/* ... cuBLAS calls ... */

cublasDestroy(handle);
```

### Step 4: Allocate and transfer data

Unlike CBLAS, which operates directly on host memory, cuBLAS operates on **device memory**. You need to allocate device arrays, copy the input data from host to device before the computation, and copy the result back afterwards.

**Allocate** device memory with `cudaMalloc`. The `Matrix` class stores `double` values (as identified in Part 1, Step 1), so the allocation size is the number of elements multiplied by `sizeof(double)`:

```c
double *d_A;
cudaMalloc(&d_A, rows * cols * sizeof(double));
```

**Copy host to device** with `cudaMemcpy`. As a reminder, the direction flag `cudaMemcpyHostToDevice` indicates the transfer direction:

```c
cudaMemcpy(d_A, A.data(), rows * cols * sizeof(double), cudaMemcpyHostToDevice);
```

**Copy device to host** after the computation using `cudaMemcpyDeviceToHost`:

```c
cudaMemcpy(A.data(), d_A, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
```

**Free** device memory with `cudaFree` when done.

```c
cudaFree(d_A);
```

**Task:** Work out the correct sizes for allocating and transferring each of the three matrices (A, B, C) in terms of `m`, `n` and `k`.

### Step 5: Error checking

CUDA runtime functions return `cudaError_t` and cuBLAS functions return `cublasStatus_t` to signal success or failure. Both should be checked after every call. `CUDA_CHECK` and `CUBLAS_CHECK` macros are provided at the top of `matmul_cublas.cu` — they print a descriptive error message and exit if a call fails:

```cpp
CUDA_CHECK(cudaMalloc(&d_A, size));
CUBLAS_CHECK(cublasDgemm(...));
```

Wrap every CUDA and cuBLAS call with the appropriate macro.

### Step 6: Implement

**Task:** Implement the full cuBLAS workflow in `matmul_cublas.cu`. The overall structure is:

1. Create the cuBLAS handle
2. Allocate device memory for A, B and C
3. Copy A and B from host to device
4. Call `cublasDgemm`
5. Copy C from device to host
6. Free device memory
7. Destroy the handle

### Step 7: Test and experiment

**Task:** Uncomment the cuBLAS section in `solve.cpp`, then build and run:

```bash
cmake -B build
cmake --build build
./build/bin/solve 512             # Small test — verify correctness
```

The output should show `Verification: PASSED`.

**Task:** Once it passes, experiment with different matrix sizes and compare the performance of the CPU and cuBLAS implementations:

```bash
./build/bin/solve 1024            # Larger matrix where CPU should still be faster
./build/bin/solve 4096            # GPU should start being faster here
./build/bin/solve 8192 4096 8192  # Non-square test where the GPU should be faster
```

Look at the reported speedups. At what size does the GPU become faster than the CPU? Try a few sizes of your own to find the crossover point on your system. If you wish, you can comment out the baseline CPU solver in `solve.cpp` and set the baseline to the blas implementation and observe the speedup with much larger matrices.
