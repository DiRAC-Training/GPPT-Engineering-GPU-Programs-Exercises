cmake_minimum_required(VERSION 3.16)
project(device_kernel_unit_test LANGUAGES CXX CUDA)

# Use an installed Catch2 package when available.
find_package(Catch2 3 QUIET CONFIG)

if(NOT Catch2_FOUND)
  # Keep the fallback source and build outside the disposable build directory.
  include(FetchContent)
  set(
    FETCHCONTENT_BASE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/Catch2"
    CACHE PATH "Persistent FetchContent directory"
  )
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.8.1
  )
  FetchContent_MakeAvailable(Catch2)
endif()

add_executable(test test.cu adders.cu)
target_link_libraries(test PRIVATE Catch2::Catch2WithMain)

# required to keep __device__ functions in separate files
set_target_properties( test PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON # This can potentially stop some optimisations: https://developer.nvidia.com/blog/building-cuda-applications-cmake/#separable_compilation

  # These properties may be required in some circumstances:
  # CUDA_RESOLVE_DEVICE_SYMBOLS ON
  # CUDA_LINK_EXECUTABLE_DEVICE_CODE ON
  # POSITION_INDEPENDENT_CODE ON
)
