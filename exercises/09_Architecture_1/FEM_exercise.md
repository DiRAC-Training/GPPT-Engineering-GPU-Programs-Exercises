# Exercise: Architecting the FEM dense solver for both CPU and GPU

## Recap and motivation

In the [Designing a GPU code module](TODO: link), you ported the FEM dense heat solver from CPU to GPU. The port replaced `solver.cpp` with `solver.cu` and changed the host allocations to `cudaMallocManaged`, leaving the project as a GPU-only build: anyone working on the CPU path could no longer build it without the CUDA toolkit, and the original CPU implementation was no longer reachable from the build.

This exercise starts from the opposite state: you are handed a **working CPU build** (the pre-porting code), and your job is to restructure it into a clean architecture and then add the GPU backend alongside the CPU one. The module 30 GPU port is available in `gpu_reference/` as a reference for the GPU-specific pieces. The end result is a single source tree that can be built either way:

```bash
make                       # CPU build (OpenMP)
make USE_GPU=1             # GPU build (CUDA + cuBLAS)

cmake -B build             # CPU build via CMake
cmake -B build -DUSE_GPU=ON
```

This puts the [Preprocessor directives](../../docs/60_Architecture_1.md#preprocessor-directives) and [Build system configuration](../../docs/60_Architecture_1.md#build-system-configuration) sections of module 60 into practice. A fully completed version is provided in the `solution/` subdirectory for reference once you are done.

## Project layout

The starting point is laid out as follows:

```shell
FEM/
├── Makefile             # CPU build complete; GPU branch is a TODO
├── CMakeLists.txt       # CPU build complete; GPU branch is a TODO
├── include/
│   ├── solver.hpp       # Declarations for matvec, axpby, dot and cg_solve
│   ├── kernels.hpp      # Empty scaffold with TODOs -- you fill it in
│   ├── utils.hpp        # DenseMatrix etc. -- worked #ifdef USE_GPU example
│   └── test.hpp         # Analytic validation declaration
├── src/
│   ├── main.cpp         # CPU driver (uses new[] / delete[])
│   ├── solver.cpp       # Full CPU solver (CG + matvec + dot + axpby)
│   ├── utils.cpp        # Assembly and boundary conditions (provided, do not modify)
│   ├── test.cpp         # Analytic validation (provided)
│   └── dat2vtk.cpp      # Optional utility: convert temperature.dat to VTK for visualisation
├── gpu_reference/       # Module 30 GPU port -- reference material only
│   ├── solver.cu        # Full GPU port (CG + kernels, managed memory)
│   └── Makefile         # GPU build makefile
└── solution/            # Fully completed reference version
```

Running `make` straight away should produce a working `build/fem_heat` executable that solves the test problem using the CPU code. Verify this first:

```bash
make
./build/fem_heat
```

You should see the CG iteration count and solve time printed, and a `temperature.dat` file written.

The shared headers live in `include/`; sources live in `src/`. The module 30 GPU port is kept in `gpu_reference/` as a reference; you will extract pieces from it in Task 2, but nothing in there is part of the build. Note that the [example layout](../../docs/60_Architecture_1.md#example-project-layout) in the module text uses a slightly different convention (headers alongside their sources inside `src/`); the same separation-of-concerns ideas apply equally here, so feel free to use whichever layout you find clearer in your own projects.

The exercise has two main tasks:

1. **Task 1 — Restructure the CPU code.** Split the linear-algebra primitives (`matvec`, `dot`, `axpby`) out of `solver.cpp` and behind a shared interface in `kernels.hpp`, so that the CG outer loop (`cg_solve`) in `solver.cpp` becomes backend-independent. The CPU build keeps working at every step.
2. **Task 2 — Add the GPU backend.** Extract the CUDA kernels from `gpu_reference/solver.cu` into a new `src/gpu_kernels.cu` behind the same interface, apply the `#ifdef USE_GPU` pattern shown in `include/utils.hpp` to `main.cpp` and `solver.cpp` so they compile for both targets, and complete the GPU branches of the `Makefile` and `CMakeLists.txt`.

We will break these tasks into smaller steps. Throughout these steps, you will be able to choose either `Make` or `CMake` to adapt the build system.

## Boundary discussion

Before starting, it is worth looking at how the existing code relates to the design points in module 60.

### Portable interface

Currently all four function declarations (`matvec`, `dot`, `axpby` and `cg_solve`) live together in `include/solver.hpp`, and `include/kernels.hpp` is an empty scaffold. In Task 1 you will split them: the three primitives move into `kernels.hpp`, leaving only `cg_solve` in `solver.hpp`.

The critical constraint on `kernels.hpp` is that it must declare its functions using only standard C++ types — `float *`, `int`, and the project's own `DenseMatrix` struct. No CUDA types (`dim3`, device pointers, stream handles), no `__global__` declarations, and no CUDA includes. This is the [Portable CPU and GPU code](../../docs/60_Architecture_1.md#portable-cpu-and-gpu-code) pattern: the GPU implementation (`src/gpu_kernels.cu`) keeps its kernel definitions and launch configuration internal, and exposes host-callable wrappers whose signatures match the CPU implementation (`src/cpu_kernels.cpp`). Because both source files satisfy the same `kernels.hpp` declarations, the build system can swap them without anything else in the project noticing.

Without this discipline the swap would not work. If `kernels.hpp` exposed a kernel signature or referenced `dim3`, including it from `src/solver.cpp` would force `solver.cpp` through a GPU compiler (in this case `nvcc`) and break any attempt at a CPU-only build.

### Managed memory and the exploratory phase

The module 30 GPU port in `gpu_reference/solver.cu` uses CUDA managed memory (`cudaMallocManaged`), and so does the worked example in `include/utils.hpp`. There is no separation between a host buffer and a device buffer: the same pointer is valid in both spaces, and the CPU branch's `new[]` allocation simply mirrors that interface on the host.  

This matches the [Early porting and prototyping](../../docs/60_Architecture_1.md#early-porting-and-prototyping) advice in the module. Managed memory is a deliberate choice for this stage of the work: it lets you restructure the project for portability — what this exercise is about — without yet committing to where the host/device boundary should sit or what data layout each side wants. If and when performance requirements push the project toward explicit `cudaMalloc` / `cudaMemcpy` and the host/device naming conventions described in the module, the source-level guards introduced in Task 2 are the natural place where that refinement will land.

## Task 1 — Restructure the CPU code

### 1a — Split the header

Open `include/solver.hpp` and `include/kernels.hpp`. The solver header currently declares four functions: the three linear-algebra primitives (`matvec`, `dot`, `axpby`) and `cg_solve`. Conceptually, the primitives and `cg_solve` play different roles: the primitives are what each backend implements, `cg_solve` is the outer loop that sits on top of them and is shared.

Move the declarations of the three primitives from `solver.hpp` into `kernels.hpp`, leaving only `cg_solve` in `solver.hpp`. The block you move is:

```cpp
// ---------------------------------------------------------------------------
// Matrix-vector product
// ---------------------------------------------------------------------------

void matvec(const DenseMatrix &A, const float *x, float *y);

// ---------------------------------------------------------------------------
// Vector operations
// ---------------------------------------------------------------------------

float dot(const float *a, const float *b, int n);
void axpby(float alpha, const float *x, float beta, float *y, int n);
```

Both headers include `utils.hpp` so `DenseMatrix` is available in either file.

After this step, `solver.hpp` declares the interface the application uses to solve; `kernels.hpp` declares the interface each backend will implement. The CPU build still works — `solver.cpp` still defines all four functions in one file, which is fine.

### 1b — Split the source

Open `src/solver.cpp`. The file contains four definitions: `matvec`, `dot`, `axpby`, and `cg_solve`. The three primitives are what differ between backends; `cg_solve` calls them but does not itself do anything CPU- or GPU-specific.

Create a new file `src/cpu_kernels.cpp` that contains only the three primitive definitions. Leave `cg_solve` in `src/solver.cpp`. Adjust the includes: `cpu_kernels.cpp` should include `kernels.hpp` (not `solver.hpp`), and `solver.cpp` should include both `solver.hpp` and `kernels.hpp` (it needs the kernels declarations to call them).

The standard library includes (`<cmath>` and `<cstring>`) should stay in whichever .cpp file uses them — they should not be added to any project headers.

### 1c — Update the build

With `solver.cpp` now missing the three primitive definitions, linking the project will fail (`undefined reference to matvec` etc.) until `cpu_kernels.cpp` is compiled and linked in too.

Update the build files in either:

#### 1c: Make

Open the **`Makefile`** file and set `KERNELS_OBJ` to `$(BUILD_DIR)/cpu_kernels.o`. The compile rule for it is already in the file — you are just telling the build to include the object in the link step.

#### 1c: CMake

Open the **`CMakeLists.txt`** file and add `target_sources(fem_heat PRIVATE src/cpu_kernels.cpp)` at the TODO marker inside the `else()` branch of the `USE_GPU` block.

### 1d — Verify

Run `make clean && make` (or reconfigure and build with CMake). You should get the same working `fem_heat` binary as before, producing the same numerical result.

This is the baseline you return to whenever Task 2 breaks something.  At any point you can `git stash` (if working with git) or comment out your Task 2 changes and confirm the CPU build is still working.

## Task 2 — Add the GPU backend

### 2a — Create `src/gpu_kernels.cu`

Open `gpu_reference/solver.cu`. The file has three distinct pieces:

- The `CublasHandle` wrapper struct and its static instance (for the cuBLAS handle used by `dot`).
- The CUDA kernels (`matvec_kernel`, `axpby_kernel`) and their host-side launchers (`matvec`, `axpby`), plus the cuBLAS-based `dot`.
- The `cg_solve` function, which you don't need — your `src/solver.cpp` now owns the CG outer loop for both backends.

Create `src/gpu_kernels.cu` containing only the first two pieces. Include `kernels.hpp` instead of `solver.hpp`, and the CUDA headers (`<cublas_v2.h>`, `<cuda_runtime.h>`). Drop `cg_solve` and the `<cmath>` / `<cstring>` includes which are no longer needed.

After this step, `src/cpu_kernels.cpp` and `src/gpu_kernels.cu` both satisfy the `kernels.hpp` interface — they are the two interchangeable implementations that the build system will pick between.

### 2b — Preprocessor guards in the shared code

Look at `include/utils.hpp`. The `DenseMatrix::allocate()` and `DenseMatrix::free()` methods, and the `<cuda_runtime.h>` include at the top of the file, are wrapped in `#ifdef USE_GPU` blocks:

```cpp
#ifdef USE_GPU
#include <cuda_runtime.h>
#endif
```

```cpp
void allocate(int n) {
    size = n;
#ifdef USE_GPU
    cudaMallocManaged(&vals, (size_t)size * size * sizeof(float));
    std::memset(vals, 0, (size_t)size * size * sizeof(float));
#else
    vals = new float[size * size]();
#endif
}
```

When `USE_GPU` is defined (which the build system will do in step 2c), the GPU branch compiles; otherwise the CPU branch does. The two branches expose the same external behaviour: a freshly allocated zero-initialised buffer of the right size.

**Apply the same pattern in two more places:**

- `src/main.cpp` — the file currently allocates the RHS and temperature vectors with `new[]` and frees them with `delete[]`. Wrap these in `#ifdef USE_GPU` blocks that use `cudaMallocManaged` / `cudaFree` (plus `std::memset` for zeroing) on the GPU path, following the `utils.hpp` pattern. Also add the guarded `<cuda_runtime.h>` include.
- `src/solver.cpp` — inside `cg_solve`, the temporary vectors `r`, `p` and `A_times_p` are currently allocated and freed with `new[]` / `delete[]`. Wrap those in the same `#ifdef` pattern. The guarded `<cuda_runtime.h>` include belongs here too.

After this step the project compiles against either backend, but the build system still has to actually set `USE_GPU` and pick the right kernel file. You will fix this in the next step.

### 2c — Complete the Makefile / CMakeLists.txt for the GPU branch

Update the build files in either:

#### 2c: Make

Open `Makefile`. Inside the `ifdef USE_GPU` block, override the CPU defaults so that:

- `CXX` becomes `$(NVCC)` — nvcc is the only compiler that can build `src/gpu_kernels.cu`, and it accepts plain C++ for the other files.
- `CXXFLAGS` sets the CUDA architecture (`-arch=$(SM_ARCH)`), passes `-DUSE_GPU` so the source-level guards activate, and points at the project includes. The OpenMP flags are not needed here.
- `LDFLAGS` links cuBLAS (`-lcublas`).
- `KERNELS_OBJ` becomes `$(BUILD_DIR)/gpu_kernels.o`.

The relevant patterns are described in [Setting preprocessor symbols from the build system](../../docs/60_Architecture_1.md#setting-preprocessor-symbols-from-the-build-system), [Conditional library linking](../../docs/60_Architecture_1.md#conditional-library-linking) and [Selecting source files](../../docs/60_Architecture_1.md#selecting-source-files). You can also look at `gpu_reference/Makefile` to see how the original module 30 build was set up.

Once complete:

```bash
make clean
make USE_GPU=1
./build/fem_heat
```

#### 2c: CMake

Open `CMakeLists.txt`. Inside the `if(USE_GPU)` block, do the following:

- `enable_language(CUDA)` and set `CMAKE_CUDA_ARCHITECTURES` (e.g. `set(CMAKE_CUDA_ARCHITECTURES 70)`).
- `find_package(CUDAToolkit REQUIRED)` so the cuBLAS imported target becomes available.
- `target_sources(fem_heat PRIVATE src/gpu_kernels.cu)` — CMake routes `.cu` files through nvcc automatically once CUDA is enabled.
- `target_compile_definitions(fem_heat PRIVATE USE_GPU)` so the source-level guards activate.
- `target_link_libraries(fem_heat PRIVATE CUDA::cublas)`.

See the same module 60 sections referenced in *2c: Make* above, inspecting the CMake side of each pattern which is shown alongside the Make side.

When complete:

```bash
cmake -B build-gpu -DUSE_GPU=ON
cmake --build build-gpu
./build-gpu/fem_heat
```

## Wrapping up

You should now have a single source tree that builds either a CPU or GPU executable from one set of files:

- `include/kernels.hpp` declares the linear-algebra primitives using only standard C++ types.
- `src/cpu_kernels.cpp` and `src/gpu_kernels.cu` each implement that interface independently — the build system links exactly one.
- `src/solver.cpp` contains the CG outer loop, shared between both backends, with `#ifdef USE_GPU` guards only at the memory allocation points.
- The `Makefile` or `CMakeLists.txt` controls which backend is compiled, which preprocessor symbols are set, and which libraries are linked.

<!-- ## Stretch — vendor selection -->

<!-- The exercise uses a single `USE_GPU` symbol, with CUDA hard-coded as the GPU backend. In a project that needed to support both NVIDIA and AMD hardware, the same pattern extends to a pair of symbols `USE_CUDA` / `USE_HIP`, with `USE_GPU` derived from whichever is active. Sketch (without writing the code) what would change in your `Makefile` and `CMakeLists.txt` to support a HIP backend alongside the existing CUDA one. The relevant module 60 discussion is in [Setting preprocessor symbols from the build system](../../docs/60_Architecture_1.md#setting-preprocessor-symbols-from-the-build-system).   

To test this, you can use [HIPIFY](TODO Link to relevant module in DiRAC HIP course) to auto-generate a HIP version that runs on AMD GPUs
-->
