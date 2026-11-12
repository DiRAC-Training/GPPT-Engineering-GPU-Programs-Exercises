# Minimal device unit tests

This example showcases two methods of testing device functions (i.e. those marked `__device__`):

1. Adding `__host__` to the function declaration. This is used to test the function `add_one` in `adders.cu`.
2. Wrapping the device function in a simple kernel. This is used to test `add_two`.

The file structure is:

- `main.cu` contains a set of test cases run in `main`, along with the wrapper kernel and a wrapper launch function that handles data movement.
- `adders.cuh` contains the declarations of the two device functions.
- `adders.cu` contains the definitions of the two device functions.
- `makefile` provides a simple makefile supporting separable compilation.

## On separable compilation

Without special flags, device function definitions and calls must be in the same file. In this example we wish to store the definitions in `adders.cu`, forward the declarations in `adders.cuh` and use the functions in the tests in `main.cu`. So, we must compile each `.cu` file with the flag `-dc` and manually link the resultant object files together.

As noted in [the section on building CUDA programs in the CUDA developer guide](https://developer.nvidia.com/blog/building-cuda-applications-cmake/#separable_compilation), this may impact performance.

## Build instructions

```bash
make && ./test
```

The `make` call is equivalent to running the commands:

```bash
nvcc -dc main.cu -o main.o
nvcc -dc adders.cu -o adders.o
nvcc main.o adders.o -o test
```
