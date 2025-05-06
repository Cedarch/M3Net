/*!
 * Modified from
 *https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
 **************************************************************************************************
 */

#include "pytorch_cpp_helper.hpp"

at::Tensor ms_deform_attn_cuda_forward(const at::Tensor &value,
                                       const at::Tensor &spatial_shapes,
                                       const at::Tensor &level_start_index,
                                       const at::Tensor &sampling_loc,
                                       const at::Tensor &attn_weight,
                                       const int im2col_step);

void ms_deform_attn_cuda_backward(
    const at::Tensor &value, const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index, const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight, const at::Tensor &grad_output,
    at::Tensor &grad_value, at::Tensor &grad_sampling_loc,
    at::Tensor &grad_attn_weight, const int im2col_step);

Tensor ms_deform_attn_forward_gpu(const Tensor &value,
                                  const Tensor &spatial_shapes,
                                  const Tensor &level_start_index,
                                  const Tensor &sampling_loc,
                                  const Tensor &attn_weight,
                                  const int im2col_step) {
  at::DeviceGuard guard(value.device());
  return ms_deform_attn_cuda_forward(value, spatial_shapes, level_start_index,
                                     sampling_loc, attn_weight, im2col_step);
}

void ms_deform_attn_backward_gpu(
    const Tensor &value, const Tensor &spatial_shapes,
    const Tensor &level_start_index, const Tensor &sampling_loc,
    const Tensor &attn_weight, const Tensor &grad_output, Tensor &grad_value,
    Tensor &grad_sampling_loc, Tensor &grad_attn_weight,
    const int im2col_step) {
  at::DeviceGuard guard(value.device());
  ms_deform_attn_cuda_backward(value, spatial_shapes, level_start_index,
                               sampling_loc, attn_weight, grad_output,
                               grad_value, grad_sampling_loc, grad_attn_weight,
                               im2col_step);
}

PYBIND11_MODULE (TORCH_EXTENSION_NAME, m) {
  m.def("ms_deform_attn_forward_gpu", &ms_deform_attn_forward_gpu,
        "ms_deform_attn_forward_gpu", py::arg("value"),
        py::arg("spatial_shapes"), py::arg("level_start_index"),
        py::arg("sampling_loc"), py::arg("attn_weight"),
        py::arg("im2col_step"));
  m.def("ms_deform_attn_backward_gpu", &ms_deform_attn_backward_gpu,
        "ms_deform_attn_backward_gpu", py::arg("value"),
        py::arg("spatial_shapes"), py::arg("level_start_index"),
        py::arg("sampling_loc"), py::arg("attn_weight"), py::arg("grad_output"),
        py::arg("grad_value"), py::arg("grad_sampling_loc"),
        py::arg("grad_attn_weight"), py::arg("im2col_step"));
}
