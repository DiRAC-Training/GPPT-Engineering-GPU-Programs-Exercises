cmake_minimum_required(VERSION 3.18)
project(signal_smooth LANGUAGES C)

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
