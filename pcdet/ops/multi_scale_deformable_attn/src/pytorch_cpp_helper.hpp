#ifndef PYTORCH_CPP_HELPER
#define PYTORCH_CPP_HELPER
#include <cuda_runtime_api.h>
#include <torch/extension.h>

#include <vector>

using namespace at;

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) \
  TORCH_CHECK(!x.device().is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CUDA_INPUT(x) \
  CHECK_CUDA(x);            \
  CHECK_CONTIGUOUS(x)
#define CHECK_CPU_INPUT(x) \
  CHECK_CPU(x);            \
  CHECK_CONTIGUOUS(x)

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

#define CHECK_CUDA_ERROR(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

#endif  // PYTORCH_CPP_HELPER
