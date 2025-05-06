#ifndef PYTORCH_CUDA_HELPER
#define PYTORCH_CUDA_HELPER

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <THC/THCAtomics.cuh>

#include <cuda.h>

using at::Half;
using at::Tensor;
using phalf = at::Half;

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

inline int GET_BLOCKS(const int N, const int num_threads = 512) {
  int optimal_block_num = (N + num_threads - 1) / num_threads;
  int max_block_num = 4096;
  return min(optimal_block_num, max_block_num);
}

// This func is used to get the no limited number of blocks and can avoid some
// potential bugs in ops
inline int GET_BLOCKS_NO_LIMIT(const int N, const int num_threads) {
  return (N + num_threads - 1) / num_threads;
}

#define __PHALF(x) (x)

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

#endif  // PYTORCH_CUDA_HELPER
