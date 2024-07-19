#pragma once

#include "tensors.cu"
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
// #include <torch/extension.h>
// #include <torch/library.h>
// #include <torch/torch.h>
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

///////////////////////
///////////////////////
///////////////////////
//////INPUT/////////
///////////////////////
///////////////////////
///////////////////////

// ok, what the bloody hell is gonna happen here? here we prcess the input
// the procecing is a multi step process. we take in a std array of 10 elements which are the input torch tensor fields.
// in the first step we get the pointerss and make a 10 elemnts tupple of them. i.e we make a input_iterator_tup
// from this tupple we make a input_ZipIterator, which is what allows us to pass them to the reduction.
// when this ZipIterator is derefenced, i.e during the reading of the data, we get a 10 elemets tupple of the inputs.
// this iterator will be made into a transform itertor in the in2out section

template <typename TP,>
using ecc_input_tup = thrust::tuple<TP, TP, TP, TP, TP, TP, TP, TP, TP, TP>;

template <typename TP>
using input_iterator_tup = ecc_input_tup<thrust::device_ptr<TP>>;
// typedef the zip_iterator of this tuple
template <typename TP>
using input_ZipIterator = thrust::zip_iterator<input_iterator_tup<TP>>;

template <typename TP>
__host__ input_ZipIterator<TP> make_input_zip_itereator(std::array<TP *, 10> inputs) {

  auto input_ptr_tup = thrust::make_tuple(thrust::device_pointer_cast<TP>(inputs[0]),
                                          thrust::device_pointer_cast<TP>(inputs[1]),
                                          thrust::device_pointer_cast<TP>(inputs[2]),
                                          thrust::device_pointer_cast<TP>(inputs[3]),
                                          thrust::device_pointer_cast<TP>(inputs[4]),
                                          thrust::device_pointer_cast<TP>(inputs[5]),
                                          thrust::device_pointer_cast<TP>(inputs[6]),
                                          thrust::device_pointer_cast<TP>(inputs[7]),
                                          thrust::device_pointer_cast<TP>(inputs[8]),
                                          thrust::device_pointer_cast<TP>(inputs[9]));

  // input_ZipIterator in_zip_iter(input_ptr_tup);

  std::cout << thrust::reduce(thrust::get<0>(input_ptr_tup), thrust::get<0>(input_ptr_tup) + 1000, TP(-1), thrust::maximum<TP>()) << std::endl;
  std::cout << thrust::reduce(thrust::get<1>(input_ptr_tup), thrust::get<1>(input_ptr_tup) + 1000, TP(-1), thrust::maximum<TP>()) << std::endl;
  std::cout << thrust::reduce(thrust::get<2>(input_ptr_tup), thrust::get<2>(input_ptr_tup) + 1000, TP(-1), thrust::maximum<TP>()) << std::endl;
  std::cout << thrust::reduce(thrust::get<3>(input_ptr_tup), thrust::get<3>(input_ptr_tup) + 1000, TP(-1), thrust::maximum<TP>()) << std::endl;
  std::cout << thrust::reduce(thrust::get<4>(input_ptr_tup), thrust::get<4>(input_ptr_tup) + 1000, TP(-1), thrust::maximum<TP>()) << std::endl;
  std::cout << thrust::reduce(thrust::get<5>(input_ptr_tup), thrust::get<5>(input_ptr_tup) + 1000, TP(-1), thrust::maximum<TP>()) << std::endl;
  std::cout << thrust::reduce(thrust::get<6>(input_ptr_tup), thrust::get<6>(input_ptr_tup) + 1000, TP(-1), thrust::maximum<TP>()) << std::endl;
  std::cout << thrust::reduce(thrust::get<7>(input_ptr_tup), thrust::get<7>(input_ptr_tup) + 1000, TP(-1), thrust::maximum<TP>()) << std::endl;
  std::cout << thrust::reduce(thrust::get<8>(input_ptr_tup), thrust::get<8>(input_ptr_tup) + 1000, TP(-1), thrust::maximum<TP>()) << std::endl;
  std::cout << thrust::reduce(thrust::get<9>(input_ptr_tup), thrust::get<9>(input_ptr_tup) + 1000, TP(-1), thrust::maximum<TP>()) << std::endl;
  std::cout << std::endl;
  return input_ZipIterator<TP>(input_ptr_tup);
}

template <typename TP>
struct ecc_input {
  const ecc_input_tup<TP> tup;
  __host__ __device__ ecc_input(const ecc_input_tup<TP> in_tup) : tup(in_tup){};

  __host__ __device__ TP gx_chosen_values() { return thrust::get<0>(tup); };
  __host__ __device__ TP gy_chosen_values() { return thrust::get<1>(tup); };
  __host__ __device__ TP Jx() { return thrust::get<2>(tup); };
  __host__ __device__ TP Jy() { return thrust::get<3>(tup); };
  __host__ __device__ TP Jxx_prime() { return thrust::get<4>(tup); };
  __host__ __device__ TP Jxy_prime() { return thrust::get<5>(tup); };
  __host__ __device__ TP Jyx_prime() { return thrust::get<6>(tup); };
  __host__ __device__ TP Jyy_prime() { return thrust::get<7>(tup); };
  __host__ __device__ TP current_level_reference_tensor_zero_mean() { return thrust::get<8>(tup); };
  __host__ __device__ TP current_level_input_tensor_warped() { return thrust::get<9>(tup); };
};

///////////////////////
///////////////////////
///////////////////////
//////OUTPUT/////////
///////////////////////
///////////////////////
///////////////////////

// ok, what the bloody hell is gonna happen here?
// the procecing is a multi step process. here we deal with the output. our output is 2 length 8 vectors and
// a symetric matrix C. these are stored in a struct called ecc_output which has the +() opertor overlloaded for the
// reduction. now when we actuly write this to the storage we need to make the matirx "full" i.e get all the elemnets
// explictly, hence the output iterator is a 3 component zip itertor, the first 2 elemnts are pointer to the Gt and Gw
// storage, the last elemnet of zeip ietrator is a transfrom ouput iterotr which takes the Mat_sym and casts it into a full mat.

// emplate <typename TP>
// using ecc_output_tup = thrust::tuple<Vec<TP, 8>, Vec<TP, 8>, Mat_Sym<TP, 8>>;

// template <typename TP>
// using output_iterator_tup = ecc_output_tup<thrust::device_ptr<TP>>;

// first we make the functor which makes the sym_mat full

template <typename TP>
using ecc_output_tup = thrust::tuple<Vec<TP, 8>, Vec<TP, 8>, Mat_Sym<TP, 8>>;

template <typename TP>
struct add_output_tup : public thrust::binary_function<ecc_output_tup<TP>, ecc_output_tup<TP>, ecc_output_tup<TP>> {
  __host__ __device__ ecc_output_tup<TP> operator()(ecc_output_tup<TP> a, ecc_output_tup<TP> b) {
    Vec<TP, 8> Gt = thrust::get<0>(a) + thrust::get<0>(b);
    Vec<TP, 8> Gw = thrust::get<1>(a) + thrust::get<1>(b);
    Mat_Sym<TP, 8> C = thrust::get<2>(a) + thrust::get<2>(b);
    return thrust::make_tuple(Gt, Gw, C);
  }
};

// using mat_ = thrust::tuple<Vec<TP, 8>, Vec<TP, 8>, Mat_Sym<TP, 8>>;
//  decalre the iteartor tupple the will be turning into a zip iterator
//  an axuliray function to extrat pointers from torch tensors. to be used in the next fucntion.

// this funxction makes a zip iteator used for the ouput. the first two elements of zip iterator are normal thrusts pointers to vectors.
//  the last elemnts of a mat_sym_2_full_iterartor (i.e a tranfrom output oterator) which casts the sym_mat to a full mat up writing.

//////////////
template <typename TP>
struct mat_sym_2_full : public thrust::unary_function<Mat_Sym<TP, 8>, Mat<TP, 8>> {

  __host__ __device__ Mat<TP, 8> operator()(Mat_Sym<TP, 8> mat_sym) {

    return Mat<TP, 8>(mat_sym);
  }
};

template <typename TP>
using mat_out_iterator = thrust::transform_output_iterator<mat_sym_2_full<TP>, thrust::device_ptr<Mat<TP, 8>>>;

template <typename TP>
using output_iterator_tup2 = thrust::tuple<thrust::device_ptr<Vec<TP, 8>>,
                                           thrust::device_ptr<Vec<TP, 8>>,
                                           mat_out_iterator<TP>>;

// typedef the zip_iterator of this tuple
template <typename TP>
using output_ZipIterator2 = thrust::zip_iterator<output_iterator_tup2<TP>>;

// this funxction makes a zip iteator used for the ouput. the first two elements of zip iterator are normal thrusts pointers to vectors.
//  the last elemnts of a mat_sym_2_full_iterartor (i.e a tranfrom output oterator) which casts the sym_mat to a full mat up writing.

template <typename TP>
__host__ output_ZipIterator2<TP> make_output_zip_itereator2(std::array<TP *, 3> outputs) {

  Vec<TP, 8> *Gt_ptr = (Vec<TP, 8> *)(outputs[0]);
  thrust::device_ptr<Vec<TP, 8>> Gt_dev_ptr = thrust::device_pointer_cast<Vec<TP, 8>>(Gt_ptr);

  Vec<TP, 8> *Gw_ptr = (Vec<TP, 8> *)(outputs[1]);
  thrust::device_ptr<Vec<TP, 8>> Gw_dev_ptr = thrust::device_pointer_cast<Vec<TP, 8>>(Gw_ptr);

  Mat<TP, 8> *C_ptr = (Mat<TP, 8> *)(outputs[2]);
  thrust::device_ptr<Mat<TP, 8>> C_dev_ptr = thrust::device_pointer_cast<Mat<TP, 8>>(C_ptr);
  mat_out_iterator<TP> mat_iter(C_dev_ptr, mat_sym_2_full<TP>());

  // auto output_ptr_tup = thrust::make_tuple(thrust::device_pointer_cast<Vec<TP, 8>>(Gt_ptr),
  //                                          thrust::device_pointer_cast<Vec<TP, 8>>(Gw_ptr),
  //                                          mat_iter);

  auto output_ptr_tup = thrust::make_tuple(Gt_ptr,
                                           Gw_ptr,
                                           mat_iter);

  // input_ZipIterator in_zip_iter(input_ptr_tup);
  return output_ZipIterator2<TP>(output_ptr_tup);
}

///////////////////////
///////////////////////
///////////////////////
//////IN_2_OUT/////////
///////////////////////
///////////////////////
///////////////////////\

// in this section we

template <typename TP>
struct ecc_in2out : public thrust::unary_function<ecc_input_tup<TP>, ecc_output_tup<TP>> {
  __host__ __device__ ecc_output_tup<TP> operator()(const ecc_input_tup<TP> input_tup) const {

    ecc_input input(input_tup);
    Vec<TP, 8> G;
    G(0) = input.gx_chosen_values() * input.Jx();
    G(1) = input.gy_chosen_values() * input.Jx();
    G(2) = -input.gx_chosen_values() * input.Jxx_prime() - input.gy_chosen_values() * input.Jxy_prime();
    G(3) = input.gx_chosen_values() * input.Jy();
    G(4) = input.gy_chosen_values() * input.Jy();
    G(5) = -input.gx_chosen_values() * input.Jyx_prime() - input.gy_chosen_values() * input.Jyy_prime();
    G(6) = input.gx_chosen_values();
    G(7) = input.gy_chosen_values();

    Vec<TP, 8> Gt = G * input.current_level_reference_tensor_zero_mean();
    Vec<TP, 8> Gw = G * input.current_level_input_tensor_warped();
    Mat_Sym<TP, 8> C = G.Outer();
    return thrust::make_tuple(Gt, Gw, C);
  }
};

/*template <typename TP>
struct mat_sym_2_full {

  __host__ __device__ Mat<TP, 8> operator(Mat_Sym<TP, 8> mat_sym) {

    return Mat<TP, 8>(mat_sym);
  }
};*/

///////////////////////
///////////////////////
///////////////////////
//////OUTPUT KEYS/////////
///////////////////////
///////////////////////
///////////////////////

// this section is a simple section desing to give each linear index in the input arrays the its corredpding page its reduced to

template <typename TP>
struct get_T_from_ind : public thrust::unary_function<TP, TP> {
  TP y;
  get_T_from_ind(TP Y) : y(Y){};

  __host__ __device__ TP operator()(TP ind) {
    // tup2 res(x,x+y);
    return ind / y;
  }
};

////////////

template <typename TP>
void ecc_reduction_impl(const std::array<TP *, 10> inputs,
                        const std::array<TP *, 3> outputs,
                        const std::array<int64_t, 2> input_size) {

  // auto torch_input_sizes = (inputs[0]).sizes();
//   std::cout << thrust::reduce(thrust::device, inputs[0], inputs[0] + 1000, TP(-1), thrust::maximum<TP>()) << std::endl;
//   std::cout << thrust::reduce(thrust::device, inputs[1], inputs[1] + 1000, TP(-1), thrust::maximum<TP>()) << std::endl;
//   std::cout << thrust::reduce(thrust::device, inputs[2], inputs[2] + 1000, TP(-1), thrust::maximum<TP>()) << std::endl;
//   std::cout << thrust::reduce(thrust::device, inputs[3], inputs[3] + 1000, TP(-1), thrust::maximum<TP>()) << std::endl;
//   std::cout << thrust::reduce(thrust::device, inputs[4], inputs[4] + 1000, TP(-1), thrust::maximum<TP>()) << std::endl;
//   std::cout << thrust::reduce(thrust::device, inputs[5], inputs[5] + 1000, TP(-1), thrust::maximum<TP>()) << std::endl;
//   std::cout << thrust::reduce(thrust::device, inputs[6], inputs[6] + 1000, TP(-1), thrust::maximum<TP>()) << std::endl;
//   std::cout << thrust::reduce(thrust::device, inputs[7], inputs[7] + 1000, TP(-1), thrust::maximum<TP>()) << std::endl;
//   std::cout << thrust::reduce(thrust::device, inputs[8], inputs[8] + 1000, TP(-1), thrust::maximum<TP>()) << std::endl;
//   std::cout << thrust::reduce(thrust::device, inputs[9], inputs[9] + 1000, TP(-1), thrust::maximum<TP>()) << std::endl;
//   std::cout << std::endl;

  input_ZipIterator<TP> input_zip_iter = make_input_zip_itereator<TP>(inputs);         // this is a zip itertor which we transfrom upon reading
  auto input_iter = thrust::make_transform_iterator(input_zip_iter, ecc_in2out<TP>()); // this is the actual data input itertor, which applys the ecc funcations to the read data

  output_ZipIterator2<TP> output_iter = make_output_zip_itereator2<TP>(outputs);
  // auto output_iter = thrust::make_transform_output_iterator(output_zip_iter , ecc_output_2_output_tup<TP>());

  get_T_from_ind get_T_from_ind_op(input_size[1]);
  auto keys_iter = thrust::make_transform_iterator(thrust::counting_iterator<int64_t>(0), get_T_from_ind_op);

  int total_input_num = input_size[0] * input_size[1];
  thrust::equal_to<int64_t> binary_pred;
  // thrust::plus<ecc_output<TP>> binary_op;
  thrust::reduce_by_key(keys_iter, keys_iter + total_input_num, input_iter, thrust::make_discard_iterator(), output_iter, binary_pred, add_output_tup<TP>());
};

// template <typename TP>
// using input_transfrom_Iterator = thrust::transform_iterator< ecc_in2out<TP>, input_ZipIterator<TP>>

// template <typename TP>
// input_transfrom_Iterator<TP> make_input_transform_iterator(
// TP* gx_chosen_values
// TP* gy_chosen_values
// TP* Jx
// TP* Jy
// TP* Jxx_prime
// TP* Jxy_prime
// TP* Jyx_prime
// TP* Jyy_prime
// TP* current_level_reference_tensor_zero_mean
// TP* current_level_input_tensor_warped){

// }