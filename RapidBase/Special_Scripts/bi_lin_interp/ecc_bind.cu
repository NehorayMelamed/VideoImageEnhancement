

#include "ecc_impl.cu"
#include <algorithm>
#include <array>
#include <bitset>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <torch/torch.h>
#include <tuple>
#include <type_traits>
#include <vector>

#include <torch/extension.h>
#include <torch/library.h>
#include <torch/torch.h>
// #include <cuda_runtime.h>

// H_matrix,
// current_level_reference_tensor_zero_mean,
// current_level_input_tensor_warped,
// Jx_chosen_values,
// Jy_chosen_values,
// gx_chosen_values,
// gy_chosen_values

using GLOB_TP = float;

auto ecc_bilinear_interpolation(const torch::Tensor &input_image,
                                const torch::Tensor &vx,
                                const torch::Tensor &vy,
                                const torch::Tensor &H_matrix,
                                const torch::Tensor &x_mat_chosen_values,
                                const torch::Tensor &y_mat_chosen_values) {

  // get the size of the number of pointes
  auto input_image_size_torch = input_image.sizes();
  auto chosen_values_size_torch = x_mat_chosen_values.sizes();
  // const std::array<int64_t, 2> chosen_values_size({*(chosen_values_size_torch.begin()), *(chosen_values_size_torch.end() - 1)});
  // const std::array<int64_t, 2> input_size({*(chosen_values_input_sizes.begin()), *(chosen_values_input_sizes.end() - 1)});

  Inputs_size inputs_size(*input_image_size_torch.begin(),
                          *(input_image_size_torch.end() - 2),
                          *(input_image_size_torch.end() - 1),
                          *(chosen_values_size_torch.end() - 1));

  // get the input image size

  Inputs_ptr<GLOB_TP> inputs_ptr(input_image.data_ptr<GLOB_TP>(),
                                 vx.data_ptr<GLOB_TP>(),
                                 vy.data_ptr<GLOB_TP>(),
                                 H_matrix.data_ptr<GLOB_TP>(),
                                 x_mat_chosen_values.data_ptr<GLOB_TP>(),
                                 y_mat_chosen_values.data_ptr<GLOB_TP>());

  // set the type for the output tensors
  auto output_tensor_options = // options for output tensor
      torch::TensorOptions()
          .dtype(x_mat_chosen_values.dtype()) // Jx.dtype()
          .layout(torch::kStrided)
          .device(torch::kCUDA)
          .requires_grad(false); /* .device(torch::kCUDA, 0)  .dtype(torch::kInt32)*/

  // create the output tensors
  torch::Tensor input_image_warped = torch::empty({inputs_size.T, inputs_size.N}, output_tensor_options);
  torch::Tensor vx_warped = torch::empty({inputs_size.T, inputs_size.N}, output_tensor_options);
  torch::Tensor vy_warped = torch::empty({inputs_size.T, inputs_size.N}, output_tensor_options);
  // torch::Tensor C = torch::empty({input_size[0], (8 + 1) * 8 / 2}, output_tensor_options);

  // make the sturct of the output tensors
  Outputs_ptr<GLOB_TP> outputs_ptr((GLOB_TP *__restrict__)input_image_warped.data_ptr<GLOB_TP>(),
                                   (GLOB_TP *__restrict__)vx_warped.data_ptr<GLOB_TP>(),
                                   (GLOB_TP *__restrict__)vy_warped.data_ptr<GLOB_TP>());

  // call the kernel
  bilinear_interpolation_ker<GLOB_TP><<<inputs_size.T, 384>>>(inputs_ptr, outputs_ptr, inputs_size);
  // ecc_reduction_impl<GLOB_TP>(inputs_ptr, outputs_ptr, input_size);
  return std::make_tuple(input_image_warped, vx_warped, vy_warped);
}


auto ecc_bilinear_interpolation_no_grad(const torch::Tensor &input_image,
                                const torch::Tensor &H_matrix,
                                const torch::Tensor &x_mat_chosen_values,
                                const torch::Tensor &y_mat_chosen_values) {

  // get the size of the number of pointes
  auto input_image_size_torch = input_image.sizes();
  auto chosen_values_size_torch = x_mat_chosen_values.sizes();
  // const std::array<int64_t, 2> chosen_values_size({*(chosen_values_size_torch.begin()), *(chosen_values_size_torch.end() - 1)});
  // const std::array<int64_t, 2> input_size({*(chosen_values_input_sizes.begin()), *(chosen_values_input_sizes.end() - 1)});

  Inputs_size inputs_size(*input_image_size_torch.begin(),
                          *(input_image_size_torch.end() - 2),
                          *(input_image_size_torch.end() - 1),
                          *(chosen_values_size_torch.end() - 1));

  // get the input image size

  Inputs_ptr_no_grad<GLOB_TP> inputs_ptr(input_image.data_ptr<GLOB_TP>(),
                                 H_matrix.data_ptr<GLOB_TP>(),
                                 x_mat_chosen_values.data_ptr<GLOB_TP>(),
                                 y_mat_chosen_values.data_ptr<GLOB_TP>());

  // set the type for the output tensors
  auto output_tensor_options = // options for output tensor
      torch::TensorOptions()
          .dtype(x_mat_chosen_values.dtype()) // Jx.dtype()
          .layout(torch::kStrided)
          .device(torch::kCUDA)
          .requires_grad(false); /* .device(torch::kCUDA, 0)  .dtype(torch::kInt32)*/

  // create the output tensors
  torch::Tensor input_image_warped = torch::empty({inputs_size.T, inputs_size.N}, output_tensor_options);
  torch::Tensor vx_warped = torch::empty({inputs_size.T, inputs_size.N}, output_tensor_options);
  torch::Tensor vy_warped = torch::empty({inputs_size.T, inputs_size.N}, output_tensor_options);
  // torch::Tensor C = torch::empty({input_size[0], (8 + 1) * 8 / 2}, output_tensor_options);

  // make the sturct of the output tensors
  Outputs_ptr<GLOB_TP> outputs_ptr((GLOB_TP *__restrict__)input_image_warped.data_ptr<GLOB_TP>(),
                                   (GLOB_TP *__restrict__)vx_warped.data_ptr<GLOB_TP>(),
                                   (GLOB_TP *__restrict__)vy_warped.data_ptr<GLOB_TP>());

  // call the kernel
  bilinear_interpolation_no_grad_ker<GLOB_TP><<<inputs_size.T, 384>>>(inputs_ptr, outputs_ptr, inputs_size);
  // ecc_reduction_impl<GLOB_TP>(inputs_ptr, outputs_ptr, input_size);
  return std::make_tuple(input_image_warped, vx_warped, vy_warped);
}

PYBIND11_MODULE(ecc_bilinear_interpolation, m) {
  m.def("ecc_bilinear_interpolation", &ecc_bilinear_interpolation, "ecc_bilinear_interpolation");
  m.def("ecc_bilinear_interpolation_no_grad", &ecc_bilinear_interpolation_no_grad, "ecc_bilinear_interpolation_no_grad");
}

// AT_DISPATCH_ALL_TYPES(Jx.dtype(), "ecc_reduction_impl", [&] { ecc_reduction_impl<scalar_t>(inputs, outputs); });
// AT_DISPATCH_GLOB_TPING_TYPES_AND_HALF(Jx.dtype(), "ecc_reduction_impl", [&] { ecc_reduction_impl<scalar_t>(inputs, outputs); });
// if ( Jx.dtype() ==  torch::GLOB_TP)
// {
//   ecc_reduction_impl<GLOB_TP>(inputs, outputs);
// }
// else if (Jx.dtype()== torch::dtype( torch::GLOB_TP))
// {
//   ecc_reduction_impl<GLOB_TP>(inputs, outputs);
// }
// else
// {
//   std::cout << "shit"  << std::endl;
// }