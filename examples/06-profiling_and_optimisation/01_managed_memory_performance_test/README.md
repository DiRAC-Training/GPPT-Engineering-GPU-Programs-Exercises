# Managed Memory vs Manual Transfers

This example measures the runtime of a simple GPU operation twice, once with managed memory and once with explicit memory. The operation: fill a large array of floats with `1.0`, add `1.0` to every element on the GPU, then check the result on the CPU.

## The two versions

**Managed memory (`runManaged`)**

The working array is created as one buffer with `cudaMallocManaged`, so both the CPU and the GPU can use the same pointer. The runtime transfers memory while it is accessed inside the kernel.

**Manual transfers (`runManual`)**

The working arrays are allocated separately as a CPU buffer with `malloc` and a GPU buffer with `cudaMalloc`. The data is explicitly copied to and from the device with `cudaMemcpy`.

## Requirements

- An NVIDIA GPU and the CUDA Toolkit (`nvcc` on your `PATH`)

## Build and run

```bash
./build.sh
./managed_example
```

Example output (RTX 3060 Laptop GPU, CUDA 13.4):

```
Manual:  correct
Managed: correct

Manual transfers:   156.72 ms
Managed memory:     333.40 ms
```

## What is being timed?

Each version is timed from start to finish with `std::chrono`, so the time covers:

- allocating memory
- filling the array on the CPU
- moving data to the GPU
- running the kernel
- moving data back and checking it
- freeing memory

## Why is managed memory slower here?

With manual copies, `cudaMemcpy` moves the whole array in one large, efficient transfer.

With managed memory, the GPU *page faults* the first time it touches each page that is still on the CPU. The driver then moves the data over in many smaller chunks. Lots of small moves are slower than one big copy. The same thing happens in reverse when the CPU reads the results.
