

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

// gx_chosen_values
// gy_chosen_values
// Jx
// Jy
// Jxx_prime
// Jxy_prime
// Jyx_prime
// Jyy_prime
// current_level_reference_tensor_zero_mean
// current_level_input_tensor_warped

using GLOB_TP = float;

auto ecc_reduction(const torch::Tensor &gx_chosen_values,
                   const torch::Tensor &gy_chosen_values,
                   const torch::Tensor &Jx,
                   const torch::Tensor &Jy,
                   const torch::Tensor &Jxx_prime,
                   const torch::Tensor &Jxy_prime,
                   const torch::Tensor &Jyx_prime,
                   const torch::Tensor &Jyy_prime,
                   const torch::Tensor &current_level_reference_tensor_zero_mean,
                   const torch::Tensor &current_level_input_tensor_warped)
{

  //  auto aaa= current_level_reference_tensor_zero_mean.data_ptr<GLOB_TP>();
  //  auto bbb= current_level_input_tensor_warped.data_ptr<GLOB_TP>();
  // std::cout << aaa << " " << thrust::reduce(thrust::device, aaa , aaa + 1000, GLOB_TP(-1), thrust::maximum<GLOB_TP>()) << std::endl;
  // std::cout << bbb << " " << thrust::reduce(thrust::device, bbb , bbb + 1000, GLOB_TP(-1), thrust::maximum<GLOB_TP>()) << std::endl;

  auto torch_input_sizes = gx_chosen_values.sizes();
  const std::array<int64_t, 2> input_size({*(torch_input_sizes.begin()), *(torch_input_sizes.end() - 1)});
  const std::array<const GLOB_TP *__restrict__, 9> inputs_ptr_arr({gx_chosen_values.data_ptr<GLOB_TP>(),
                                                                   gy_chosen_values.data_ptr<GLOB_TP>(),
                                                                   Jx.data_ptr<GLOB_TP>(),
                                                                   Jy.data_ptr<GLOB_TP>(),
                                                                   Jxx_prime.data_ptr<GLOB_TP>(),
                                                                   Jxy_prime.data_ptr<GLOB_TP>(),
                                                                   Jyx_prime.data_ptr<GLOB_TP>(),
                                                                   Jyy_prime.data_ptr<GLOB_TP>(),
                                                                   current_level_input_tensor_warped.data_ptr<GLOB_TP>()});

  const float *current_level_reference_tensor_zero_mean_ptr = current_level_reference_tensor_zero_mean.data_ptr<GLOB_TP>();

  ecc_inputs_ptr ecc_inputs(inputs_ptr_arr);

  auto output_tensor_options = // options for output tensor
      torch::TensorOptions()
          .dtype(Jx.dtype()) // Jx.dtype()
          .layout(torch::kStrided)
          .device(torch::kCUDA)
          .requires_grad(false); /* .device(torch::kCUDA, 0)  .dtype(torch::kInt32)*/

  torch::Tensor G = torch::empty({input_size[0], input_size[1], 8}, output_tensor_options);
  torch::Tensor Gt = torch::empty({input_size[0], 8,1}, output_tensor_options);
  torch::Tensor Gw = torch::empty({input_size[0], 8,1}, output_tensor_options);
  torch::Tensor C = torch::empty({input_size[0], 8, 8}, output_tensor_options);
  // torch::Tensor C = torch::empty({input_size[0], (8 + 1) * 8 / 2}, output_tensor_options);

  ecc_outputs_ptr ecc_outputs((Vec<GLOB_TP, 8> *__restrict__)(G.data_ptr<GLOB_TP>()),
                              (Vec<GLOB_TP, 8> *__restrict__)(Gt.data_ptr<GLOB_TP>()),
                              (Vec<GLOB_TP, 8> *__restrict__)(Gw.data_ptr<GLOB_TP>()),
                              (Mat<GLOB_TP, 8> *__restrict__)(C.data_ptr<GLOB_TP>()));

  // std::cout << input_size[0] << " " << input_size[1]  << std::endl;
  ecc_reduction_ker<<<input_size[0], 256>>>(ecc_inputs, current_level_reference_tensor_zero_mean_ptr, ecc_outputs, input_size);
  // ecc_reduction_impl<GLOB_TP>(inputs_ptr, outputs_ptr, input_size);  
  return std::make_tuple(G, Gt, Gw, C);
}

PYBIND11_MODULE(ecc_reduction, m)
{
  m.def("ecc_reduction", &ecc_reduction, "ecc_reduction");
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