cmake_minimum_required(VERSION 3.18)
project(signal_smooth LANGUAGES C)

# CPU version using FFTW3
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3 REQUIRED fftw3)

add_executable(cpu signal_smooth_cpu.c)
target_include_directories(cpu PRIVATE ${FFTW3_INCLUDE_DIRS})
target_link_libraries(cpu ${FFTW3_LIBRARIES} m)

# GPU version using cuFFTW (optional, requires CUDA)
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    add_executable(gpu signal_smooth_gpu.c)
    target_link_libraries(gpu CUDA::cufftw CUDA::cufft m)
else()
    message(STATUS "CUDA not found - skipping GPU version")
endif()
