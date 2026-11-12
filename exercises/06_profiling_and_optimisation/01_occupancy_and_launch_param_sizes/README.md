# Launch parameters and occupancy

This example is designed to show the impact of kernel launch parameters on the occupancy of that kernel.

`main.cu` runs one axpy kernel (`x = 2*x + y`) with a single thread per element.
The block size is passed as the only argument; the grid is derived from it, so every run does the same work.

## Exercise 1: Getting used to the code

**Spend some time looking at the code**

Make sure the code builds:

```sh
./build.sh
```

Now try running with a given **block size**:

```sh
./exe <block size>

# e.g.
./exe 32
```

## Exercise 2: Profiling with Nsight Compute

Let's run Nsight Compute on the program directly in the command line.
We will use the `-k` argument to filter the Compute output to only include the kernel of interest, in this case the `axpy_kernel`.

```sh
ncu -k axpy_kernel exe 32
```

**Identify and note down the duration of the kernel.**

*Hint*: The `Duration` metric in the `Speed of Light Throughhput` table is the execution time of the kernel.

**Now identify the `Expected occupancy` associated with block size.**

## Exercise 3: Changing occupancy with block size

You're about to change the block size and see how it changes the expected occupancy. Based on what you already know about occupancy, **can you predict what will happen to the `Expected occupancy` as you change the block size?**

**Run Nsight Compute on the kernel again, with a range of input block sizes.**

```sh
ncu -k axpy_kernel exe <num>
```

The block size should be a multiple of the number of the warp size (i.e. 32).
We recommend testing the block sizes 32, 64, 128, 256, 768 and 1024.
Note that the block size must fit within one SM, and so has a theoretical largest size.
For an A100, this size is 1024.

You should write down the input parameter, and the measured values of `Block Limit Warps` and `Expected occupancy`. **How do your measurements compare to your earlier predictions?**