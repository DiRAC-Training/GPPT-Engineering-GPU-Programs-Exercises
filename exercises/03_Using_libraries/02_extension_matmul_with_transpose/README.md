# Exercise 1: GEMM Porting to BLAS Libraries

This exercise demonstrates porting a naive matrix multiplication (GEMM) to BLAS libraries.

## Structure

```bash
matrix.hpp           # Matrix class with contiguous storage
matmul.hpp           # Function declarations
matmul_cpu.cpp       # CPU implementations: serial + parallel OpenMP (provided)
matmul_blas.cpp      # BLAS implementation (your task)
matmul_cublas.cu     # cuBLAS implementation (your task)
solve.cpp            # Main driver (provided)
solution/matmul_blas.cpp       # Reference solution for BLAS (two approaches)
solution/matmul_cublas.cu      # Reference solution for cuBLAS (two approaches)
```

## The task

1. Implement `matmul_blas()` in `matmul_blas.cpp` using a BLAS library
2. Implement `matmul_cublas()` in `matmul_cublas.cu` using cuBLAS
3. Uncomment the corresponding sections in `solvep` to test your implementations

## Documentation

**CBLAS (CPU):**

Note the `CBLAS_ORDER` parameter which accepts `CblasRowMajor`

- [Netlib CBLAS cblas_dgemm](https://www.netlib.org/lapack/explore-html/dc/d18/cblas__dgemm_8c.html)
- [Intel oneMKL cblas_dgemm](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/cblas-gemm-001.html) 

**cuBLAS (GPU):**

Note the Data Layout section for cuBLAS, as well as the GEMM API documentation.  Is our code in row-major or column-major? What options do you have to ensure the code and the library use the same data layout?

- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/index.html#data-layout)
- [cublasDgemm reference](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm)

**Key insight:** CBLAS provides `CblasRowMajor` to handle C/C++ row-major arrays directly. cuBLAS does not - it always uses column-major (Fortran) layout. Think about what this means mathematically: a row-major matrix viewed as column-major is its transpose. How can we efficiently deal with this?

## Building

```bash
cmake -B build
cmake --build build
```

## Running

To execute the code you have modified, run:

```bash
./build/bin/solve                # Reference solution square 512 x 512 (Useful for testing, but too small to run faster on a GPU)
./build/bin/solve 4096           # Square 4096 x 4096 (About the size where the GPU code will be faster)
./build/bin/solve 8192 4096 8192 # A(8192 x 4096) * B(4096 x 8192) = C(8192x8192) (Large enough for the GPU code to be noticeably faster)
```

If you wish to execute the provided solution, run:

```bash
./build/bin/solution 4096
./build/bin/solution 8192 4096 8192
```

## Verification

When correctly implemented, the output will show:
`Verification: PASSED` (the library result matches the hand-written cpu implementation).

## Hints (if stuck)

**Hint 1: Understanding your data layout**

C/C++ stores 2D arrays in **row-major** order (rows are contiguous in memory). Fortran and most BLAS libraries use **column-major** order (columns are contiguous).

Before calling any BLAS function, ask: what layout does the library expect? CBLAS is designed for C and accepts a `CblasRowMajor` parameter. Fortran BLAS and cuBLAS expect column-major.

**Hint 2: Understanding leading dimensions (lda, ldb, ldc)**

The "leading dimension" is the stride (distance in memory) between consecutive rows (row-major) or columns (column-major). For a contiguous MxN matrix:

- **Row-major**: `ld = N` (number of columns) - stride to the next row
- **Column-major**: `ld = M` (number of rows) - stride to the next column

For GEMM computing C = A × B where A is MxK, B is KxN, C is MxN:

| Layout       | lda | ldb | ldc |
|--------------|-----|-----|-----|
| Row-major    |  K  |  N  |  N  |
| Column-major |  M  |  K  |  M  |

**Hint 3: CBLAS solution**

The first parameter to `cblas_dgemm` lets you specify the memory layout. Use `CblasRowMajor` and the rest is straightforward.

**Hint 4: cuBLAS - using transpose flags**

You can use `CUBLAS_OP_T` to tell cuBLAS to transpose the inputs. This fixes the input interpretation, but cuBLAS still **outputs** in column-major. Make sure your result is in the expected form - you may need to transpose C afterwards.

**Hint 5: cuBLAS - alternative approach**

Instead of adapting the library to match your data, you can adapt your code to match the library.

If your row-major A is seen as A^T by cuBLAS, and B is seen as B^T, what happens if you swap the order and compute B * A instead of A * B?

You get: B^T * A^T = (A * B)^T

Stored column-major, but read as row-major, the transpose "cancels out" and you get A * B - no extra transpose needed.
