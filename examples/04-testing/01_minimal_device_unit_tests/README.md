# Minimal device unit tests

This example showcases two methods of testing device functions (i.e. those marked `__device__`):

1. Adding `__host__` to the function declaration. This is used to test the function `add_one` in `adders.cu`.
2. Wrapping the device function in a simple kernel. This is used to test `add_two`.

The file structure is as follows:

- `test.cu` contains the actual test run by Catch2, along with the wrapper kernel and a wrapper launch function that handles data movement.
- `adders.cuh` contains the declarations of the two device functions.
- `adders.cu` contains the definitions of the two device functions.
- `CMakeLists.txt` provides a simple CMake build file that downloads Catch2 if it is not already available and compiles 

## On separable compilation

Without special flags, device function definitions and calls must be in the same file. In this example we wish to store the definitions in `adders.cu`, forward the declarations in `adders.cuh` and use the functions in the tests in `test.cu`. So, we must instruct the CUDA build system to allow this with the CMake target option `CUDA_SEPARABLE_COMPILATION`.

As noted in [the section on building CUDA programs in the CUDA developer guide](https://developer.nvidia.com/blog/building-cuda-applications-cmake/#separable_compilation), this may impact performance.

## Build instructions

The minimal device unit test uses [Catch2](https://github.com/catchorg/Catch2). The provided CMake configuration uses a system installation when available and otherwise downloads and builds Catch2 automatically.

```bash
cmake -B build
cmake --build build
./build/test
```
