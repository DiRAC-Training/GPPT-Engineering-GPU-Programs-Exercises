cmake_minimum_required(VERSION 3.16)
project(kernel_testing LANGUAGES CXX CUDA)

include(CTest)

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

if(DEFINED catch2_SOURCE_DIR)
  list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
elseif(DEFINED Catch2_DIR)
  list(APPEND CMAKE_MODULE_PATH "${Catch2_DIR}")
endif()
include(Catch)

add_subdirectory(src)


