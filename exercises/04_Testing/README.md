# Testing GPU code - exercise

In this exercise you will write a series of unit tests for a simple `__global__` kernel and a `__device__` function.

## The exercise code

First, let's look at the code we will be using for the exercise.

### Project layout

Familiarise yourself with the layout of the project.
The files you will edit live at the top level; the `solution/` directory mirrors this layout with a complete reference implementation.

```text
matvec_exercise/
├── CMakeLists.txt        # top-level build configuration
├── include/
│   ├── kernels.hpp       # host-callable declarations
│   └── kernels.cuh       # device function declaration
├── src/
│   ├── CMakeLists.txt    # build configuration for the targets
│   ├── kernels.cu        # the code under test and its wrappers
│   ├── main.cpp          # a small driver showing run_matvec with managed memory
│   └── test.cu           # all unit tests (compiled as CUDA)
└── solution/             # complete reference implementation, mirroring the above
```

The functions you test are declared in `include/kernels.hpp` and defined in `src/kernels.cu`.
You write all your tests in `src/test.cu`, which is compiled as CUDA.

### The code to be tested

The functions that you will be writing unit tests for can be found in [`src/kernels.cu`](./src/kernels.cu).
If you have worked with the [Conjugate Gradient exercise](), a similar version of the `matvec_kernel` may be familiar to you from its solution.

If you don't recognise the solution, don't worry!
The kernel carries out a [matrix-vector product](https://mathinsight.org/matrix_vector_multiplication) calculation $y = A * x$, where $A$ is a matrix and $x$ a vector.
The output of this calculation, $y$, is a vector whose elements are the sum of the matrix elements in that row multiplied respectively by each of the vector's elements.
This can be represented by the diagram in [`matvec.pdf`](../../../../figures/matvec.pdf).

`matvec_kernel` parallelises this calculation over each of the elements in the output vector (i.e. an independent dot product of each row of the matrix and the input vector).  

In this kernel, we represent our 2D matrix $A$ using a flattened 1D array that strides over the columns (row-major). In order to calculate the correct index for an element in $A$ from its 2D position in the matrix, `matvec_kernel` calls the device function `get_index`.
It can be useful to factor out index calculations like this into individual device functions so that if the indexing of data objects changes (e.g. to change from row-major to column-major), the index calculation can be changed in one place rather than many throughout the code.

### The testing code

The tests use [Catch2](https://github.com/catchorg/Catch2). The provided CMake configuration uses a system installation when available and otherwise downloads and builds Catch2 automatically.

To configure and compile the code:

```bash
cmake -B build
cmake --build build
```

Run the tests directly with `./build/test`, or through CTest:

```bash
ctest --test-dir build --output-on-failure
```

Some scaffolding is provided to set up the Catch2 tests using `TEST_CASE` and a helper function, `require_output_matches` which uses Catch2 to test array elements with a provided tolerance, `epsilon`.

Note: the separable compilation CMake target property is only enabled for the `matvec_tests` target.

Now that you're familiar with the simple kernels involved and the build environment, you can begin the exercises.

## **Exercise** Adding unit tests for the kernels

In this exercise, you will use the various approaches described in the written content to test `__global__` kernels and `__device__` functions.
For more fundamental concepts on testing, please refer to [DiRAC Training Academy - Foundation HPC Skills: Ensuring Quality and Reliability in Code](https://training-academy.dirac.ac.uk/course/view.php?id=28#module-810)

You should write your tests in the [`src/test.cu`](./src/test.cu) file provided unless otherwise directed.
You will need to check that the tests run correctly in order to complete the exercise.
You may do this any way that works for you, e.g. running them stand-alone or as part of a test framework.

### Part 1: Testing `matvec_kernel`

First, you will concentrate on testing the `matvec_kernel` function in `kernels.cu`.

Let us consider what tests would be appropriate for this function.
The first case uses the identity matrix:

$$
A =
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

Any vector multiplied by the identity matrix will return the same vector.
We can therefore write a test for the `matvec_kernel` function that takes the identity matrix and any input vector as arguments, and checks that the produced output matches the initial vector.

The second case uses a full dense matrix.
An example set of values you can use to test the function are:

$$
A =
\begin{bmatrix}
1 & 2 & 3 & 4 \\
5 & 6 & 7 & 8 \\
9 & 10 & 11 & 12 \\
13 & 14 & 15 & 16
\end{bmatrix}
\qquad
x =
\begin{bmatrix}
0 \\
1 \\
-2 \\
3
\end{bmatrix}
$$

This will give the solution:

$$
y =
\begin{bmatrix}
8 \\
16 \\
24 \\
32
\end{bmatrix}
$$

Both cases — identity and dense — are provided for you in a shared data table at the top of [`src/test.cu`](./src/test.cu): `A[2][n*n]` holds the two matrices (case `0` the identity, case `1` the dense matrix), alongside the shared input vector `x` and the expected outputs `y_expected[2][n]`. Defining the inputs once, in one place, means every test below checks exactly the same data.

Each test runs against both cases using Catch2's `GENERATE(0, 1)` macro, which re-runs the test body once per case and reports the two independently: if the identity case fails you still see the dense result — which a single loop over the cases would not give you, as it would stop at the first failure. An `INFO("case: …")` line records which case any failure came from.

Now let's look at three ways to drive the kernel from a test.

#### Part 1.1: Using an existing wrapper function

In the kernels file [`src/kernels.cu`](./src/kernels.cu) there is a wrapper function `run_matvec` that calls `matvec_kernel`.
This wrapper function allows us to call the kernel from outside the compilation unit with no additional change to the code infrastructure.
As there is minimal additional work done in this function, testing `run_matvec` is more or less equivalent to testing the `matvec_kernel` directly with fixed launch parameters.
As such, you will first create a test for this function.  

In the test file, [`src/test.cu`](./src/test.cu), you will see a single `TEST_CASE` for this part, with a `SKIP` line and `TODO` comments. It is already set up to run against both cases via `GENERATE(0, 1)`.
Remove the `SKIP` line and fill in the `TODO`s to write your test of `run_matvec`.  

The input matrix `A[c]`, vector `x`, and expected output `y_expected[c]` for the current case are provided in the shared table, so you do not need to define them yourself.
Because `run_matvec` does no memory management, you must make its inputs accessible to both the host and device: allocate managed memory and copy `A[c]` and `x` into it.
This test is in essence a Catch2 parametrised version of the unit test example described in the Unit testing section of the course [TODO: link to Moodle course of 70_Testing Unit testing section]. You can use that example directly as a reference of managed allocation implementation using `cudaMallocManaged`.

With the managed inputs ready, call `run_matvec`, then check the output it produces against `y_expected[c]`.
You may use the provided `require_output_matches` helper, which compares two arrays element-wise to within a tolerance `epsilon`, or implement your own check.  

Because the test is parametrised with `GENERATE(0, 1)`, passing it means both the identity and dense cases have succeeded — there is no separate dense `TEST_CASE` to complete.

#### Part 1.2: Writing a custom thin wrapper function

The `run_matvec` function launches the `matvec_kernel`, but does not prepare the data for the device in any way. This means that in our test routine we must handle the memory management ourselves.
Ideally, our test suite would be agnostic to the code's implementation, i.e. would not contain any CUDA-specific implementations itself.
In order to avoid this, your next exercise will be to write a new wrapper function for `matvec_kernel` that handles the data movement required for the kernel itself.

The function definition for `custom_matvec_wrapper` has been provided for you in [`src/kernels.cu`](./src/kernels.cu).
You should write your custom wrapper therein. You will need to allocate the device-accessible memory here, copy the host data to it, and then call the `run_matvec` function (or launch `matvec_kernel` directly) to launch the kernel.
You may wish to reuse the same managed memory approach from the previous exercise (noting you will still need to copy the host data to the managed memory arrays), or you may choose to use explicit device memory and transfers, `cudaMalloc` and `cudaMemcpy`, such as in the provided solution.  

Once you have implemented your wrapper, complete the `custom_matvec_wrapper` `TEST_CASE` provided in [`src/test.cu`](./src/test.cu) to test your implementation. It is very similar to the previous exercise, but must ensure the test only manages host memory and does not have to be aware of the kernel's execution target: the host arrays (`A[c]` and `x`) are passed straight to the wrapper.

#### Part 1.3: Launching the kernel directly

The wrappers above are convenient, but `run_matvec` fixes the launch configuration internally — it always uses a block size of 256. To test the kernel under a *different* configuration you have to launch it yourself; for example, to check it still produces the right answer with a single thread per block.

This is also the most direct way to test a kernel: no wrapper, just the launch inside the test body. So the test can see the kernel — defined in another translation unit ([`src/kernels.cu`](./src/kernels.cu)) — `matvec_kernel` is forward-declared for you above the `TEST_CASE`. Fill in the `TODO`s: allocate managed memory and copy `A[c]` and `x` into it, choose a block size (try `1`) and the matching grid size, launch `matvec_kernel<<<grid, block>>>`, synchronise, and check the result against `y_expected[c]`.

Launching a kernel defined in another translation unit like this relies on separable compilation, which is already enabled for the test target. We return to this in Part 2.3.

### Part 2: Testing `get_index`

For the next part of the exercise, you will be testing the `get_index` device function.

The logic that this function implements is simple: it takes the width, $n$, of the matrix $A$, and the locations, $i$ and $j$, of a particular element in that matrix, and converts them into the index of the element as stored in a 1D array.
We have factored this calculation out of the `matvec_kernel` code so that changes to the indexing of the data structure can be made here directly, propagating them everywhere that we attempt to access it.

To test this function, we will use a selection of example input variables for which we know the output index value.
Example values you can use for the test are:

$$
\begin{array}{llll}
n = 64,   & i = 5,  & j = 7,   & \text{output} = 327 \\
n = 128,  & i = 64, & j = 100, & \text{output} = 8292 \\
n = 256,  & i = 0,  & j = 50,  & \text{output} = 50 \\
n = 1024, & i = 37, & j = 0,   & \text{output} = 37888
\end{array}
$$

As this function is decorated only with `__device__`, it is compiled for the device and cannot be called from host code. Further, without separable compilation, it can only be called by device code in the same translation unit.
Testing the function's logic might, therefore, be simple enough, but accessing the function will be trickier.
Let's look at some ways that we can test this function.

#### Part 2.1: Using the `__host__` decorator to access device code

The simplest way to test a device function is to add the `__host__` decorator to allow it to be called from the host.
Add this decorator now.

You will then need to implement a function that calls the new host code with the desired inputs.
An example declaration `run_get_index` has been provided for you in [`src/kernels.cu`](./src/kernels.cu), for which you will need to fill out the implementation.

You will then need to write the test for this function.
You may use the `get_index from host code` `TEST_CASE` provided in [`src/test.cu`](./src/test.cu) for your implementation.
The test should follow the same structure as those in part 1: construct your input parameters, call the function you are testing, then test the output.
Make sure your test passes before continuing.

Note that in this particular case, the `get_index` function is a strong candidate to be a `__device__ __host__` decorated function.
If we wish to change the indexing of our data structures – to check the performance of different indexing, for example – it makes sense to have this single function accessible from both host and device.
We can imagine other device functions that are less appropriate to be host callable, and therefore this is not necessarily a solution that makes sense for every device function.

The next part of the exercise will address this.

#### Part 2.2: Writing a thin wrapper function

Another way to test device-only code is to write a thin wrapper `__global__` kernel whose only purpose is to call the device code.
We saw an example of this in the written content based around a fictitious device function `device_op`.
You may use this example as a reference whilst carrying out this exercise, in which you will write your own wrapper and test.

Firstly, you will need to write your wrapper kernel.
A space has been left for you to write the wrapper in [`src/kernels.cu`](./src/kernels.cu) under the `TODO: Exercise 2.2`.
As with the example wrapper, this kernel should contain minimal code: all it needs to do is call the device function using the correct inputs, and write the result to an output buffer.

Once you've written the wrapper, you will need to implement the test.
Launching a global function directly requires a `.cu` file — as you already did for the direct kernel launch in Part 1.3 — so write your test in the provided [`src/test.cu`](./src/test.cu) file, in the `get_index through a device wrapper` `TEST_CASE` included.
The test itself will take the same form as the previous exercise: construct your input parameters, call the function you are testing, then test the output.

Remember that you will need to forward declare your wrapper function in the test's file for the compilation unit to correctly identify it.

#### Part 2.3: Using a `.cuh` file to expose the device function

This method of testing our device code works, but we have had to alter our source code by adding additional wrapper functions into it in order to accommodate testing.
If we want to keep our separation of testing and production code, we can declare our device function in a compilation unit header (`.cuh`) file, just like standard C/C++ function.
If we include this header file in our test file, we can then call the device function from our test code.
You could equally forward-declare `get_index` in the test, as we did for `matvec_kernel` in Part 1.3 — but a header keeps a single source of truth for the declaration and allows other kernels to call the device function without further forward declaration.
The final exercise for this section will be to write such a declaration for the `get_index` function.

Firstly, you should write your device function declaration in the provided [`include/kernels.cuh`](./include/kernels.cuh) file.

You will then need to add the `kernels.cuh` file into the [`src/test.cu`](./src/test.cu) include list, so that your tests recognise the declaration.

Now, you can move the wrapper function from the previous part of this exercise into the test file.
The test you wrote in part 2.2 will now run the local version of the wrapper function, and should still run correctly.
Make sure that this is the case!

Remember that in order for the compilation units to share device code, you will need to enable the `CUDA_SEPARABLE_COMPILATION` CMake target property, or pass the `-dc` flag when using `nvcc` either directly or through Make.
For this exercise this is done for you, but you should bear this flag in mind if you are aiming to do such testing in the future.
