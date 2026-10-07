# Example: FFT-based signal smoothing with FFTW and cuFFTW

This example implements a simple signal processing task: smoothing a noisy sine wave using FFT-based convolution with a Gaussian kernel. The CPU version uses FFTW3, a widely-used FFT library. The GPU version uses cuFFTW, NVIDIA's FFTW-compatible interface to cuFFT.

## The CPU version (FFTW3)

The CPU implementation (`signal_smooth_cpu.c`) uses FFTW3 to perform the convolution:

1. Generate a noisy test signal
2. Create a Gaussian smoothing kernel
3. Transform both to frequency domain using `fftw_plan_dft_r2c_1d`
4. Multiply pointwise in frequency space
5. Transform back using `fftw_plan_dft_c2r_1d`

Memory is allocated using FFTW's own allocators (`fftw_alloc_real`, `fftw_alloc_complex`) which ensure optimal alignment for vectorised operations.

## Porting to GPU (cuFFTW)

NVIDIA provides [cuFFTW](https://docs.nvidia.com/cuda/cufft/#fftw-interface-to-cufft) as a compatibility layer that implements the FFTW3 API using cuFFT internally. This allows existing FFTW code to run on GPUs with minimal changes.

**Header change:**

```c
// CPU version
#include <fftw3.h>

// GPU version
#include <cufftw.h>
```

**Memory allocation:**

cuFFTW does not provide `fftw_alloc_*` functions. Standard `malloc` must be used instead:

```c
// CPU version
double *signal = fftw_alloc_real(N);
fftw_complex *freq = fftw_alloc_complex(N/2 + 1);

// GPU version
double *signal = (double *)malloc(sizeof(double) * N);
fftw_complex *freq = (fftw_complex *)malloc(sizeof(fftw_complex) * (N/2 + 1));
```

Similarly, `fftw_free` becomes `free`.

**Build system:**

The CMakeLists.txt shows the linking differences:

```cmake
# CPU version
pkg_check_modules(FFTW3 REQUIRED fftw3)
target_link_libraries(cpu ${FFTW3_LIBRARIES} m)

# GPU version
find_package(CUDAToolkit REQUIRED)
target_link_libraries(gpu CUDA::cufftw CUDA::cufft m)
```

## Building and running

```bash
cmake -B build
cmake --build build

# Run CPU version
./build/cpu

# Run GPU version
./build/gpu
```

Both versions output timing information. On a system with a modern GPU, the GPU version should complete the FFT operations significantly faster for large problem sizes ($N = 2^{22}$ in this example).

### A note on portability:

The code above was ported to GPUs without portability in mind, as the GPU port no longer works on the CPU.  Approaches for keeping the CPU and GPU version in the same code-base as well as cross-portability with other vendor libraries will be discussed in [TODO: Link to relevant section]().

## Visualising the output

A gnuplot script is provided to visualise the smoothing effect:

```bash
gnuplot plot_signal.gp
```

This produces `signal_smooth.png` showing the original noisy signal and the smoothed result.
