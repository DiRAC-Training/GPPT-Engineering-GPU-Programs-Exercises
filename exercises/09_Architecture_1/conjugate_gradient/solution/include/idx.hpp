// Row-major index into an n x n matrix. Host only: the GPU backend defines its
// own __device__ equivalent in src/gpu_kernels.cu.
inline int idx(int i, int j, int n) { return i * n + j; }
