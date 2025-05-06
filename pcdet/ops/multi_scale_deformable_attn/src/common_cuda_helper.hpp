#ifndef VOYDET_OPS_CSRC_KERNELS_COMMON_CUDA_HELPER_HPP_
#define VOYDET_OPS_CSRC_KERNELS_COMMON_CUDA_HELPER_HPP_

#include <algorithm>

#include <cuda.h>

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

#endif  // VOYDET_OPS_CSRC_KERNELS_COMMON_CUDA_HELPER_HPP_
