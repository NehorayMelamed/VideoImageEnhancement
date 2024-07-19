#pragma once

#include "tensors.cu"
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <torch/torch.h>
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

template <typename TP>
using ecc_input_tup = thrust::tuple<TP, TP, TP, TP, TP, TP, TP, TP, TP, TP>;

template <typename TP>
using input_iterator_tup = ecc_input_tup<thrust::device_ptr<TP>>;
// typedef the zip_iterator of this tuple
template <typename TP>
using input_ZipIterator = thrust::zip_iterator<input_iterator_tup<TP>>;

template <typename TP>
__host__ __device__ input_ZipIterator<TP> make_input_zip_itereator(std::array<torch::Tensor, 10> inputs)
{
  auto get_dev_ptr_from_inputs = [&](int ind)
  { return thrust::device_pointer_cast<TP>(inputs[ind].data_ptr<TP>()); };

  auto input_ptr_tup = thrust::make_tuple(get_dev_ptr_from_inputs(0),
                                          get_dev_ptr_from_inputs(1),
                                          get_dev_ptr_from_inputs(2),
                                          get_dev_ptr_from_inputs(3),
                                          get_dev_ptr_from_inputs(4),
                                          get_dev_ptr_from_inputs(5),
                                          get_dev_ptr_from_inputs(6),
                                          get_dev_ptr_from_inputs(7),
                                          get_dev_ptr_from_inputs(8),
                                          get_dev_ptr_from_inputs(9));

  // input_ZipIterator in_zip_iter(input_ptr_tup);
  return input_ZipIterator<TP>(input_ptr_tup);
}

template <typename TP>
struct ecc_input
{
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

/*
template <typename TP>
struct mat_sym_2_full {

  __host__ __device__ Mat<TP, 8> operator(Mat_Sym<TP, 8> mat_sym) {

    return Mat<TP, 8>(mat_sym);
  }
};

// next we make a transform output itertor that applys mat_sym_2_full to a Mat_Sym<TP,8>
template <typename TP>
using mat_sym_2_full_iterartor = thrust::transform_output_iterator<mat_sym_2_full<TP>, thrust::device_vector<Mat<TP, 8>>>;

// decalre the iteartor tupple the will be turning into a zip iterator
template <typename TP>
using output_iterator_tup = thrust::tuple<thrust::device_ptr<Vec<TP, 8>>,
                                          thrust::device_ptr<Vec<TP, 8>>,
                                          mat_sym_2_full_iterartor<TP>>;

// typedef the zip_iterator of this tuple
template <typename TP>
using output_ZipIterator = thrust::zip_iterator<output_iterator_tup<TP>>;

// an axuliray function to extrat pointers from torch tensors. to be used in the next fucntion.
template <typename TP>
thrust::device_ptr<TP> get_dev_ptr_from_outputs(int ind, std::array<torch::Tensor, 3> outputs) {

  return thrust::device_pointer_cast<TP>(outputs[ind].data_ptr<TP>());
};


//this funxction makes a zip iteator used for the ouput. the first two elements of zip iterator are normal thursts pointers to vectors.
// the last elemnts of a mat_sym_2_full_iterartor (i.e a tranfrom output oterator) which casts the sym_mat to a full mat up writing.

template <typename TP>
__host__ __device__ output_ZipIterator make_outpur_zip_itereator(std::array<torch::Tesnor, 3> outputs) {

  auto output_ptr_tup = thrust::make_tuple(get_dev_ptr_from_outputs<Vec<TP, 8>>(0, outputs),
                                           get_dev_ptr_from_outputs<Vec<TP, 8>>(1, outputs),
                                           mat_sym_2_full_iterartor(mat_sym_2_full,get_dev_ptr_from_outputs<Mat<TP, 8>>(2, outputs) ));

  // input_ZipIterator in_zip_iter(input_ptr_tup);
  return output_ZipIterator(output_ptr_tup);
}
*/
template <typename TP>
using ecc_output_tup = thrust::tuple<Vec<TP, 8>, Vec<TP, 8>, Mat_Sym<TP, 8>>;

// decalre the iteartor tupple the will be turning into a zip iterator
template <typename TP>
using output_iterator_tup = thrust::tuple<thrust::device_ptr<Vec<TP, 8>>,
                                          thrust::device_ptr<Vec<TP, 8>>,
                                          thrust::device_ptr<Mat<TP, 8>>>;

// typedef the zip_iterator of this tuple
template <typename TP>
using output_ZipIterator = thrust::zip_iterator<output_iterator_tup<TP>>;

// an axuliray function to extrat pointers from torch tensors. to be used in the next fucntion.
template <typename TP>
thrust::device_ptr<TP> get_dev_ptr_from_outputs(int ind, std::array<torch::Tensor, 3> outputs)
{
  TP* raw_ptr =  outputs[ind].data_ptr<TP>();
  thrust::device_ptr< TP > thr_ptr = thrust::device_pointer_cast<TP> (raw_ptr);
  return thr_ptr;
};

// this funxction makes a zip iteator used for the ouput. the first two elements of zip iterator are normal thursts pointers to vectors.
//  the last elemnts of a mat_sym_2_full_iterartor (i.e a tranfrom output oterator) which casts the sym_mat to a full mat up writing.

template <typename TP>
__host__ __device__ output_ZipIterator<TP> make_output_zip_itereator(std::array<torch::Tensor, 3> outputs)
{

  auto output_ptr_tup = thrust::make_tuple(get_dev_ptr_from_outputs<Vec<TP, 8>>(0, outputs),
                                           get_dev_ptr_from_outputs<Vec<TP, 8>>(1, outputs),
                                           get_dev_ptr_from_outputs<Mat<TP, 8>>(2, outputs));

  // input_ZipIterator in_zip_iter(input_ptr_tup);
  return output_ZipIterator<TP>(output_ptr_tup);
}

// this struct stores the intermideite resutlts of the output during the reduction. i.e it is
template <typename TP>
struct ecc_output
{
   Vec<TP, 8> Gt;
   Vec<TP, 8> Gw;
   Mat_Sym<TP, 8> C;
   
  __host__ __device__ ecc_output<TP>() : Gt(Vec<TP, 8>(0)), Gw(Vec<TP, 8>(0)), C(Mat_Sym<TP, 8>(0)) {};
  //__host__ __device__ ecc_output(ecc_output_tup<TP> in_tup) : Gt(in_tup.get<0>()), Gw(in_tup.get<1>()), C(in_tup.get<2>()){};
  __host__ __device__ ecc_output<TP>(Vec<TP, 8> Gt_in, Vec<TP, 8> Gw_in, Mat_Sym<TP, 8> C_in) : Gt(Gt_in),Gw(Gw_in),C(Gt_in){};

  // ecc_output(Mat3_Sym<TP> in_mat, Vec2<TP> in_vec) : tup(thrsut::make_tuple(in_mat, int_vec)){};
  //__host__ __device__ ecc_output(ecc_output_tup in_tup) : tup(in_tup){};

  __host__ __device__ ecc_output<TP> operator+(const ecc_output<TP> a) const 
  {

    Vec<TP, 8> Gt_out = Gt + a.Gt;
    Vec<TP, 8> Gw_out = Gw + a.Gw;
    Mat_Sym<TP, 8> C_out = C + a.C;
    return ecc_output<TP>(Gt_out, Gw_out, C_out);
  }

    __host__ __device__ ecc_output<TP> operator+=(const ecc_output<TP> a)
  {

    Gt()+= a.Gt;
    Gc()+= a.Gc;
    C()+= a.C;
    return *this;
  }

  //   __host__ __device__ ecc_output& operator=(ecc_output a)
  // {
  //   tup=ecc_output_tup<TP>(a.Gt(),a.Gw(),a.C());
  //   return *this;
  // }
};

/*
template <typename TP>
struct ecc_output
{
  ecc_output_tup<TP> tup;
  __host__ __device__ ecc_output() : tup(ecc_output_tup<TP>(Vec<TP, 8>(0), Vec<TP, 8>(0), Mat_Sym<TP, 8>(0)  )){};
  __host__ __device__ ecc_output(ecc_output_tup<TP> in_tup) : tup(in_tup){};
  __host__ __device__ ecc_output(Vec<TP, 8> Gt_in, Vec<TP, 8> Gw_in, Mat_Sym<TP, 8> C_in) : tup(thrust::make_tuple(Gt_in, Gw_in, C_in)){};

  __host__ __device__ Vec<TP, 8> Gt() { return thrust::get<0>(tup); };
  __host__ __device__ Vec<TP, 8> Gw() { return thrust::get<1>(tup); };
  __host__ __device__ Mat_Sym<TP, 8> C() { return thrust::get<2>(tup); };

  // ecc_output(Mat3_Sym<TP> in_mat, Vec2<TP> in_vec) : tup(thrsut::make_tuple(in_mat, int_vec)){};
  //__host__ __device__ ecc_output(ecc_output_tup in_tup) : tup(in_tup){};

  __host__ __device__ ecc_output operator+(ecc_output a)
  {

    Vec<TP, 8> Gt_out = Gt() + a.Gt();
    Vec<TP, 8> Gw_out = Gw() + a.Gw();
    Mat_Sym<TP, 8> C_out = C() + a.C();
    return ecc_output(Gt_out, Gw_out, C_out);
  }

    __host__ __device__ ecc_output operator+=(ecc_output a)
  {

    Vec<TP, 8> Gt_out = Gt() + a.Gt();
    Vec<TP, 8> Gw_out = Gw() + a.Gw();
    Mat_Sym<TP, 8> C_out = C() + a.C();
    tup=ecc_output_tup<TP>(Gt_out, Gw_out, C_out);
    return *this;
  }

  //   __host__ __device__ ecc_output& operator=(ecc_output a)
  // {
  //   tup=ecc_output_tup<TP>(a.Gt(),a.Gw(),a.C());
  //   return *this;
  // }
};
*/
/*
template <typename TP>
using ecc_full_mat_output_tup = thrust::tuple<Vec<TP, 8>, Vec<TP, 8>, Mat<TP, 8>>;

template <typename TP>
struct make_ecc_full_mat_out_tup : public thrust::unary_function<ecc_output<TP>, ecc_full_mat_output_tup<TP>> {
  __host__ __device__
      ecc_output<TP>
      operator()(ecc_input<TP> input) const {
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
    return ecc_output(Gt, Gw, C);
  }
};*/

///////////////////////
///////////////////////
///////////////////////
//////IN_2_OUT/////////
///////////////////////
///////////////////////
///////////////////////\

// in this section we

template <typename TP>
struct ecc_in2out : public thrust::unary_function<ecc_input_tup<TP>, ecc_output<TP>>
{
  __host__ __device__ ecc_output<TP> operator()(const ecc_input_tup<TP> input_tup) const
  {

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
    return ecc_output(Gt, Gw, C);
  }
};

/*template <typename TP>
struct mat_sym_2_full {

  __host__ __device__ Mat<TP, 8> operator(Mat_Sym<TP, 8> mat_sym) {

    return Mat<TP, 8>(mat_sym);
  }
};*/

template <typename TP>
struct ecc_output_2_output_tup : public thrust::unary_function<ecc_output<TP>, ecc_output_tup<TP>>
{

  __host__ __device__ ecc_output_tup<TP> operator()(const ecc_output<TP> dat_in) const
  {
    Mat<TP, 8> mat_full(dat_in.C);
    return thrust::make_tuple(dat_in.Gt, dat_in.Gw, mat_full);
  }
};
///////////////////////
///////////////////////
///////////////////////
//////OUTPUT KEYS/////////
///////////////////////
///////////////////////
///////////////////////

// this section is a simple section desing to give each linear index in the input arrays the its corredpding page its reduced to

template <typename TP>
struct get_T_from_ind : public thrust::unary_function<TP, TP>
{
  TP y;
  get_T_from_ind(TP Y) : y(Y){};

  __host__ __device__ TP operator()(TP ind)
  {
    // tup2 res(x,x+y);
    return ind / y;
  }
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