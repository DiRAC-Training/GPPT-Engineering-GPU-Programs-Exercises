cmake_minimum_required(VERSION 3.18)
project(signal_smooth LANGUAGES C)

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
