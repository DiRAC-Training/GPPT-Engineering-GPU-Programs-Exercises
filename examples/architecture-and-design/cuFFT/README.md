# Example: Restructuring FFT-based signal smoothing for portability

## Introduction

In the similar libraries example, we presented two FFT-based signal smoothing programs: one based on the CPU-based FFTW, and a GPU port based on cuFFTW. We ended up with two separate files: `signal_smooth_gpu.c` from the initial code in `signal_smooth_cpu.c`. These files can be found in `original/`.  To summarise, the files differ in exactly three respects:

1. The header included (`<fftw3.h>` vs `<cufftw.h>`)
2. Complex array allocation in `convolve_fft` (`fftw_alloc_complex` vs `malloc`)
3. Real array allocation in `main` and corresponding deallocation (`fftw_alloc_real`/`fftw_free` vs `malloc`/`free`)

How might we combine these into a single source that can target either GPU or CPU?

This example walks through two approaches to restructuring these files into a single codebase that supports both targets:

- **Single file:** uses preprocessor guards in a single source file to switch between the CPU and GPU code paths at compile time.
- **Multi-file:** separates the FFT logic into dedicated files, with the build system selecting the correct implementation.

## Single source with preprocessor guards

**File:** `single_source/signal_smooth.c`

The three differences listed above are all small, self-contained pieces of code: a different header, a different allocator, a different deallocator. These differences are small enough that we can choose to keep all code in a single file and use `#ifdef` guards to switch between the two code paths at compile time based on the symbol `USE_GPU`.

**Read the file to find the five uses of `USE_GPU`.**

---

**Do you foresee any problems with maintain a code written in this way?**

<details>
<summary>Show answer</summary>

The main disadvantages of this method are:

- It's easy to forget to wrap each relevant call in the `USE_GPU` guard, and the resultant error may not be useful.
- The code becomes cluttered with details of allocation / deallocation. Supporting different hardware or libraries could make this even worse.
- Instead of writing allocations with one line, you must write the entire block of portable allocations, slowing down development time.

</details>

### Building with Make

The Makefile does two things:

1. passes the `USE_GPU` preprocessor symbol to the compiler when appropriate, and
2. links against the correct FFT library.

This is done in the following block:

```makefile
ifdef USE_GPU
  CFLAGS += -DUSE_GPU -I$(NVHPC_ROOT)/cuda/include -I$(NVHPC_ROOT)/math_libs/include
  LIBS += -L$(NVHPC_ROOT)/cuda/lib64 -L$(NVHPC_ROOT)/math_libs/lib64 -lcufftw -lcufft
else
  LIBS += -lfftw3
endif
```

Now we can build the CPU and GPU versions with:

```bash
make
make USE_GPU=1
```

### Building with CMake

The `CMakeLists.txt` follows the same logic as the Makefile but uses CMake's built-in mechanisms for option handling, package discovery and target configuration. The `option()` command exposes `USE_GPU` as a configurable build setting, defaulting to `OFF`. When enabled, `find_package(CUDAToolkit)` locates the CUDA installation and `target_compile_definitions` passes the `USE_GPU` symbol to the compiler — the CMake equivalent of the `-DUSE_GPU` flag in the Makefile. The library linking uses CMake's imported targets (`CUDA::cufftw`, `CUDA::cufft`) rather than raw linker flags:

```cmake
option(USE_GPU "Build with GPU support using cuFFTW" OFF)

add_executable(signal_smooth signal_smooth.c)
target_link_libraries(signal_smooth m)

if(USE_GPU)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(signal_smooth PRIVATE USE_GPU)
    target_link_libraries(signal_smooth CUDA::cufftw CUDA::cufft)
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFTW3 REQUIRED fftw3)
    target_include_directories(signal_smooth PRIVATE ${FFTW3_INCLUDE_DIRS})
    target_link_libraries(signal_smooth ${FFTW3_LIBRARIES})
endif()
```

The build commands mirror the Make workflow; the only difference is how the option is passed:

```bash
# CPU version
cmake -B build
cmake --build build

# GPU version
cmake -B build -DUSE_GPU=ON
cmake --build build
```

### Removing boilerplate with abstractions

**File:** `single_source/signal_smooth_refactored_malloc.c`

One of the clear disadvantages of the above approach is the amount of clutter and boilerplate it introduces whenever an allocation or deallocation is used. We can refactor this into **abstractions** of the allocations and deallocations using C macros like so:

```cpp
#ifdef USE_GPU
#define ALLOC_COMPLEX(N) (fftw_complex *)malloc(sizeof(fftw_complex) * (N))
#define ALLOC_REAL(N) (double *)malloc(sizeof(double) * N)
#define FREE(PTR) free(PTR)
#else
#define ALLOC_COMPLEX(N) fftw_alloc_complex(N)
#define ALLOC_REAL(N) fftw_alloc_real(N)
#define FREE(PTR) fftw_free(PTR)
#endif
```

**Compare `single_source/signal_smooth_refactored_malloc.c` and `single_source/signal_smooth.c` to see the difference this abstraction makes.**

---

**What are some obvious disadvantages to this abstraction approach?**

<details>
<summary>Show answer</summary>

In this case, there's relatively little disadvantage. A macro is a good way to implement this abstraction. In C++, one could use this macro implementation, lambdas, or something more complex. This abstraction works well precisely because all allocations and deallocations involved in this example have practically identical interfaces that we can hide behind a common one.

</details>

## Multi-source project structure

**Directory:** `multi_source/`

Rather than toggling between code paths in a single file, this approach separates each target's implementation into its own source file. The build system selects the correct one. This gives cleaner separation of concerns (each file deals only with its own target) and it scales better as implementations diverge. However we will see it has a few obvious downsides: the project is more complex than necessary for this particular code, and we retain some duplication between the two versions.

### Structure

```shell
multi_source/
├── CMakeLists.txt
├── Makefile
└── src/
    ├── main.c               # Shared application code
    ├── convolve_fft.h       # Common interface
    ├── convolve_fft.c       # FFTW implementation
    └── convolve_fft.c       # cuFFTW implementation
```

### The interface

`convolve_fft.h` declares a single function representing the common interface shared by the two versions:

```c
void convolve_fft(double *signal, double *kernel, double *result, int n);
```

### The implementations

Each target provides its own `convolve_fft.c` conforming to that interface. `convolve_fft_cpu.c` includes `<fftw3.h>` and uses `fftw_alloc_complex` and `fftw_free`. `convolve_fft_gpu.c` includes `<cufftw.h>` and uses `malloc` and `free`. Both files include `convolve_fft.h` and provide the same function signature.

### The main file

`main.c` includes `convolve_fft.h` and calls `convolve_fft()` without knowing which implementation will be linked.

Note that `main.c` uses `malloc`/`free` for the signal, kernel and result arrays rather than `fftw_alloc_real`/`fftw_free`. This is because `main.c` does not include any FFT headers, so those functions are not available. This is one clear disadvantage to this approach and suggests the common function interface is too inflexible for this problem.

### Make & CMake changes

The `Makefile` and `CMakeLists.txt` are similar to previous examples but are explicit about the sources. Can you find where in these files the build systems are selecting the correct source files?

## Balancing the techniques

The illustrated techniques have obvious downsides: preprocessor guards don't always scale well, while file restructuring to expose common interfaces can become complex or overly restrictive if implemented prematurely. You should use your own judgement to choose between these techniques and balance the downsides. You will find yourself refactoring and redesigning the boundaries between implementations as your code evolves.

Don't be afraid to keep things architecturally simple as you develop a code. It may be valuable to wait until an obvious architecture reveals itself before making a concrete decision on how a code should be structured.
