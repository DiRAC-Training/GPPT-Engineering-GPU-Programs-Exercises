# Example: Restructuring FFT-based signal smoothing for portability

## Introduction

In the [Libraries module](TODO: link), we ported an FFT-based signal smoothing program from the CPU-based FFTW to the GPU-based cuFFTW. We ended up with two separate files: `signal_smooth_gpu.c` from the initial code in `signal_smooth_cpu.c`. These files can be found in `original/`.  To summarise, the files differ in exactly three respects:

1. The header included (`<fftw3.h>` vs `<cufftw.h>`)
2. Complex array allocation in `convolve_fft` (`fftw_alloc_complex` vs `malloc`)
3. Real array allocation in `main` and corresponding deallocation (`fftw_alloc_real`/`fftw_free` vs `malloc`/`free`)

With so much duplication between the files, it's natural for us to consider how we might combine these into a single source that can target either GPU or CPU. This example walks through two approaches to restructuring these files into a single codebase that supports both targets, demonstrating the architectural techniques discussed in the module.

- **Single file:** uses preprocessor guards in a single source file to switch between the CPU and GPU code paths at compile time.
- **Multi-file:** separates the FFT logic into dedicated files, with the build system selecting the correct implementation.

## Single source with preprocessor guards

**Directory:** `single_source/`

The three differences listed above are all small, self-contained pieces of code: a different header, a different allocator, a different deallocator. These differences are small enough that we can choose to keep all code in a single file and use `#ifdef` guards to switch between the two code paths at compile time.

The merged file `signal_smooth.c` introduces a single preprocessor symbol, `USE_GPU`, and wraps each of the three differences in an `#ifdef` block:

```c
/* 1. Header selection */
#ifdef USE_GPU
#include <cufftw.h>
#else
#include <fftw3.h>
#endif
```

```c
/* 2. Complex array allocation in convolve_fft */
#ifdef USE_GPU
    fftw_complex *sig_freq = (fftw_complex *)malloc(sizeof(fftw_complex) * (n/2 + 1));
    fftw_complex *ker_freq = (fftw_complex *)malloc(sizeof(fftw_complex) * (n/2 + 1));
#else
    fftw_complex *sig_freq = fftw_alloc_complex(n/2 + 1);
    fftw_complex *ker_freq = fftw_alloc_complex(n/2 + 1);
#endif
```

```c
/* 3. Real array allocation and deallocation in main */
#ifdef USE_GPU
    double *signal = (double *)malloc(sizeof(double) * N);
    /* ... */
#else
    double *signal = fftw_alloc_real(N);
    /* ... */
#endif

    /* code body */

#ifdef USE_GPU
    free(signal);
    /* ... */
#else
    fftw_free(signal);
    /* ... */
#endif

```

Everything else is identical between the two targets.

This approach works well here precisely because the guarded blocks are small and few. If the differences between targets were more extensive, these blocks would grow and become difficult to read and maintain, at which point refactoring the code would then become more appealing.

### Building with Make

The Makefile needs to do two things: pass the `USE_GPU` preprocessor symbol to the compiler when GPU support is requested, and link against the correct library. Following [Setting preprocessor symbols from the build system](../../docs/60_Architecture_1.md#setting-preprocessor-symbols-from-the-build-system), the `ifdef` block checks whether the developer has set `USE_GPU` on the command line. If so, it appends `-DUSE_GPU` to the compiler flags, which is equivalent to adding `#define USE_GPU` at the top of the source file. The library flags are set accordingly — cuFFTW and cuFFT for the GPU path, FFTW3 for the CPU path — following the [Conditional library linking](../../docs/60_Architecture_1.md#conditional-library-linking) pattern.

Unlike CMake, Make has no built-in package discovery, so the GPU include and library paths must be provided explicitly. Here we use `NVHPC_ROOT`, which is set by the HPC module on most systems with the NVIDIA HPC SDK installed:

```makefile
ifdef USE_GPU
  CFLAGS += -DUSE_GPU -I$(NVHPC_ROOT)/cuda/include -I$(NVHPC_ROOT)/math_libs/include
  LIBS += -L$(NVHPC_ROOT)/cuda/lib64 -L$(NVHPC_ROOT)/math_libs/lib64 -lcufftw -lcufft
else
  LIBS += -lfftw3
endif
```

The build rule itself is a single line that compiles and links `signal_smooth.c` with the selected flags and libraries. The entire Makefile is fewer than 20 lines — the conditional block is the only GPU-specific logic:

```bash
# CPU version (FFTW) — no symbol required, defaults to CPU
make

# GPU version (cuFFTW) — pass USE_GPU to trigger the ifdef block
make USE_GPU=1
```

### Building with CMake

The CMakeLists.txt follows the same logic as the Makefile but uses CMake's built-in mechanisms for option handling, package discovery and target configuration. The `option()` command exposes `USE_GPU` as a configurable build setting, defaulting to `OFF`. When enabled, `find_package(CUDAToolkit)` locates the CUDA installation and `target_compile_definitions` passes the `USE_GPU` symbol to the compiler — the CMake equivalent of the `-DUSE_GPU` flag in the Makefile. The library linking uses CMake's imported targets (`CUDA::cufftw`, `CUDA::cufft`) rather than raw linker flags, as described in [Conditional library linking](../../docs/60_Architecture_1.md#conditional-library-linking):

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

The CPU path uses the tool `pkg-config` to locate FFTW3, which is the standard discovery mechanism for this library. The build commands mirror the Make workflow — the only difference is how the option is passed:

```bash
# CPU version
cmake -B build
cmake --build build

# GPU version
cmake -B build -DUSE_GPU=ON
cmake --build build
```

## Multi-source project structure

**Directory:** `multi_source/`

Rather than toggling between code paths in a single file, this approach separates each target's implementation into its own source file. The build system selects the correct one. This gives cleaner separation of concerns — each file deals only with its own target — and scales better as implementations diverge. However we will see it has a few obvious downsides: the project is more complex than necessary for this particular code, and we retain some duplication between the two versions.

### Structure

The first step, following [Separating GPU code into dedicated files](../../docs/60_Architecture_1.md#separating-gpu-code-into-dedicated-files), is to pull the FFT-specific code out of the main application and into dedicated files. The `convolve_fft` function — the only part that differs between targets — gets its own source file under `cpu/` and `gpu/`, while everything else stays in `main.c`:

```shell
multi_source/
├── CMakeLists.txt
├── Makefile
└── src/
    ├── main.c               # Shared application code
    ├── convolve_fft.h       # Common interface
    ├── cpu/
    │   └── convolve_fft.c   # FFTW implementation
    └── gpu/
        └── convolve_fft.c   # cuFFTW implementation
```

### The interface

With the FFT code separated out, the shared application code needs a way to call it without knowing which implementation will be linked. `convolve_fft.h` declares a single function:

```c
void convolve_fft(double *signal, double *kernel, double *result, int n);
```

The signature uses only standard C types: no `fftw_complex`, no `fftw_plan`, no CUDA types. This is the boundary between the shared application code and the target-specific FFT implementation.

### The implementations

Each target provides its own `convolve_fft.c` behind that interface. `cpu/convolve_fft.c` includes `<fftw3.h>` and uses `fftw_alloc_complex` and `fftw_free`. `gpu/convolve_fft.c` includes `<cufftw.h>` and uses `malloc` and `free`. Both files include `convolve_fft.h` and provide the same function signature — the rest of the application cannot tell which is linked.

### The main file

`main.c` contains all the shared code: signal generation, kernel creation, timing, output. It includes `convolve_fft.h` and calls `convolve_fft()` without knowing which implementation will be linked. Crucially, `main.c` has no FFT library includes or types — it compiles with a standard C compiler regardless of the target. This is the [build isolation](../../docs/60_Architecture_1.md#separating-gpu-code-into-dedicated-files) benefit described in the module: only the target-specific files require the corresponding library headers and toolchain.

Note that `main.c` uses `malloc`/`free` for the signal, kernel and result arrays rather than `fftw_alloc_real`/`fftw_free`. This is because `main.c` does not include any FFT headers, so those functions are not available. For this example the FFTW-aligned allocators are not required (they provide SIMD-aligned memory which FFTW can exploit for certain plan types, but `FFTW_ESTIMATE` plans do not benefit from this).

### Building with Make

Unlike the single-source approach, the Makefile no longer needs to set preprocessor symbols — there are no `#ifdef` blocks in the code. Instead, it selects which source file to compile alongside `main.c`, following the [Selecting source files](../../docs/60_Architecture_1.md#selecting-source-files) pattern. The `ifdef` block picks the correct implementation file and links the corresponding library. As with the single-source Makefile, the GPU include and library paths are provided explicitly via `NVHPC_ROOT`:

```makefile
ifdef USE_GPU
  CFLAGS += -I$(NVHPC_ROOT)/cuda/include -I$(NVHPC_ROOT)/math_libs/include
  SOURCES = src/main.c src/gpu/convolve_fft.c
  LIBS += -L$(NVHPC_ROOT)/cuda/lib64 -L$(NVHPC_ROOT)/math_libs/lib64 -lcufftw -lcufft
else
  SOURCES = src/main.c src/cpu/convolve_fft.c
  LIBS += -lfftw3
endif
```

The build commands are the same as the single-source case:

```bash
# CPU version
make

# GPU version
make USE_GPU=1
```

### Building with CMake

The CMakeLists.txt follows the same logic. The executable is initially created with only `main.c`, then `target_sources` adds the correct implementation file based on the `USE_GPU` option. This is the CMake equivalent of selecting source files at build time, as described in [Selecting source files](../../docs/60_Architecture_1.md#selecting-source-files):

```cmake
option(USE_GPU "Build with GPU support using cuFFTW" OFF)

add_executable(signal_smooth src/main.c)
target_link_libraries(signal_smooth m)

if(USE_GPU)
    find_package(CUDAToolkit REQUIRED)
    target_sources(signal_smooth PRIVATE src/gpu/convolve_fft.c)
    target_link_libraries(signal_smooth CUDA::cufftw CUDA::cufft)
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFTW3 REQUIRED fftw3)
    target_sources(signal_smooth PRIVATE src/cpu/convolve_fft.c)
    target_include_directories(signal_smooth PRIVATE ${FFTW3_INCLUDE_DIRS})
    target_link_libraries(signal_smooth ${FFTW3_LIBRARIES})
endif()
```

Note the key difference from the single-source CMakeLists.txt: there is no `target_compile_definitions` call. The `USE_GPU` symbol is only used by the build system to choose which file to compile — it never reaches the source code.

```bash
# CPU version
cmake -B build
cmake --build build

# GPU version
cmake -B build -DUSE_GPU=ON
cmake --build build
```

## Running

Both approaches produce the same `signal_smooth` executable:

```bash
./signal_smooth          # or ./build/signal_smooth if built with CMake
```

The program prints timing information and writes `signal_output.dat`. The gnuplot script in the parent directory can be used to visualise the result:

```bash
gnuplot ../plot_signal.gp
```

## Balancing the techniques

Both the illustrated techniques have obvious downsides: preprocessor guards don't scale well and file restructuring to expose common interfaces can become complex if implemented prematurely. Using both these techniques throughout your code will balance the downsides and allow you to produce a codebase that is both multi-architecture, and suitably maintainable. You will find yourself refactoring and redesigning the boundaries between implementations as your code evolves. Don't be afraid to keep things architecturally simple as you develop a code. It may be valuable to wait until an obvious architecture reveals itself before making a concrete decision on how a code should be structured.

## Conclusion

Here we've explored using two different methods of combining CPU and GPU codes into a single, multi-architecture codebase. Using the preprocessor allows precise control of which parts of the code are compiled, so a developer can wrap architecture-specific code in `#ifdef` blocks and a user can specify the desired architecture at compile-time. Separating architecture-specific code into files can help manage growing complexity in a code and may separate implementations in a useful way. 

In this example, the preprocessor guards are certainly enough, allowing the different versions to coexist in the same code without the code becoming too complex. This is mainly because the two FFT libraries are designed to have a very similar interface. Were we to use an FFT library with a significantly different interface, splitting the implementations further would be much more valuable. In this scenario, our second design, with different implementations in different files, would make much more sense.

These techniques can be combined to allow you as a developer to craft and evolve an appropriate design for your codebase.
