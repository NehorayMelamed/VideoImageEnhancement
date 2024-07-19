
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
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <torch/torch.h>
#include <tuple>
#include <vector>
#include <type_traits>
#include "ecc_impl.cu"

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

template <typename TP>
void ecc_reduction_impl(std::array<torch::Tensor, 10> inputs,
                        std::array<torch::Tensor, 3> outputs)
{

  auto torch_input_sizes = (inputs[0]).sizes();

  input_ZipIterator<TP> input_zip_iter = make_input_zip_itereator<TP>(inputs);         // this is a zip itertor which we transfrom upon reading
  auto input_iter = thrust::make_transform_iterator(input_zip_iter, ecc_in2out<TP>()); // this is the actual data input itertor, which applys the ecc funcations to the read data

  output_ZipIterator<TP> output_iter = make_output_zip_itereator2<TP>(outputs);
  // auto output_iter = thrust::make_transform_output_iterator(output_zip_iter , ecc_output_2_output_tup<TP>());

  get_T_from_ind get_T_from_ind_op(*(torch_input_sizes.end() - 1));
  auto keys_iter = thrust::make_transform_iterator(thrust::counting_iterator<int64_t>(0), get_T_from_ind_op);

  int total_input_num = (*(torch_input_sizes.end() - 1)) * (*(torch_input_sizes.begin()));
  thrust::equal_to<int64_t> binary_pred;
  // thrust::plus<ecc_output<TP>> binary_op;
  thrust::reduce_by_key(keys_iter, keys_iter + total_input_num, input_iter, thrust::make_discard_iterator(), output_iter, binary_pred, add_output_tup<TP>());
};

auto ecc_reduction(torch::Tensor &gx_chosen_values,
                   torch::Tensor &gy_chosen_values,
                   torch::Tensor &Jx,
                   torch::Tensor &Jy,
                   torch::Tensor &Jxx_prime,
                   torch::Tensor &Jxy_prime,
                   torch::Tensor &Jyx_prime,
                   torch::Tensor &Jyy_prime,
                   torch::Tensor &current_level_reference_tensor_zero_mean,
                   torch::Tensor &current_level_input_tensor_warped)
{

  auto torch_input_sizes = gx_chosen_values.sizes();
  std::array<int64_t, 2> input_size = {*(torch_input_sizes.begin()), *(torch_input_sizes.end() - 1)};

  std::array<torch::Tensor, 10> inputs = {gx_chosen_values,
                                          gy_chosen_values,
                                          Jx,
                                          Jy,
                                          Jxx_prime,
                                          Jxy_prime,
                                          Jyx_prime,
                                          Jyy_prime,
                                          current_level_reference_tensor_zero_mean,
                                          current_level_input_tensor_warped};

  auto output_tensor_options = // options for output tensor
      torch::TensorOptions()
          .dtype(Jx.dtype())
          .layout(torch::kStrided)
          .device(torch::kCUDA)
          .requires_grad(false); /* .device(torch::kCUDA, 0)  .dtype(torch::kInt32)*/

  torch::Tensor Gt = torch::empty({input_size[0]}, output_tensor_options);
  torch::Tensor Gw = torch::empty({input_size[0]}, output_tensor_options);
  torch::Tensor C = torch::empty({input_size[0], 3, 3}, output_tensor_options);
  // torch::Tensor C = torch::empty({input_size[0], (8 + 1) * 8 / 2}, output_tensor_options);

  std::array<torch::Tensor, 3> outputs = {Gt, Gt, C};

  // AT_DISPATCH_ALL_TYPES(Jx.dtype(), "ecc_reduction_impl", [&] { ecc_reduction_impl<scalar_t>(inputs, outputs); });
  // AT_DISPATCH_FLOATING_TYPES_AND_HALF(Jx.dtype(), "ecc_reduction_impl", [&] { ecc_reduction_impl<scalar_t>(inputs, outputs); });
  // if ( Jx.dtype() ==  torch::float)
  // {
  //   ecc_reduction_impl<float>(inputs, outputs);
  // }
  // else if (Jx.dtype()== torch::dtype( torch::double))
  // {
  //   ecc_reduction_impl<double>(inputs, outputs);
  // }
  // else
  // {
  //   std::cout << "shit"  << std::endl;
  // }

  ecc_reduction_impl<double>(inputs, outputs);
  return std::make_tuple(Gt, Gw, C);
}

PYBIND11_MODULE(ecc_reduction, m)
{
  m.def("ecc_reduction", &ecc_reduction, "ecc_reduction");
  // m.def("dummy", &dummy, "dummy desc");
  // m.def("sup_dude", &sup_dude, "suo dude desccc");
}
