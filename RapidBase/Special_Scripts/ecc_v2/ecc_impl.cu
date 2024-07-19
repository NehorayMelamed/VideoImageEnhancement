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
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>
#include <tuple>
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

template <typename TP = float>
struct ecc_inputs_ptr
{

  static constexpr int n_inputs=9;
  std::array<const TP *__restrict__, n_inputs> inputs_ptrs;

  __host__ __device__ ecc_inputs_ptr(const std::array<const TP *__restrict__, n_inputs> inputs) : inputs_ptrs(inputs){};

  __host__ __device__ TP gx_chosen_values() const { return *inputs_ptrs[0]; };
  __host__ __device__ TP gy_chosen_values() const { return *inputs_ptrs[1]; };
  __host__ __device__ TP Jx() const { return *inputs_ptrs[2]; };
  __host__ __device__ TP Jy() const { return *inputs_ptrs[3]; };
  __host__ __device__ TP Jxx_prime() const { return *inputs_ptrs[4]; };
  __host__ __device__ TP Jxy_prime() const { return *inputs_ptrs[5]; };
  __host__ __device__ TP Jyx_prime() const { return *inputs_ptrs[6]; };
  __host__ __device__ TP Jyy_prime() const { return *inputs_ptrs[7]; };
  __host__ __device__ TP current_level_input_tensor_warped() const { return *inputs_ptrs[8]; };

  __host__ __device__ ecc_inputs_ptr<TP> &operator++()
  {

#pragma unroll
    for (int ind = 0; ind < n_inputs ; ind++)
    {
      inputs_ptrs[ind]++;
    }
    return *this;
  };

  template <typename TP2>
  __host__ __device__ ecc_inputs_ptr<TP> &operator+=(TP2 add_val)
  {

#pragma unroll
    for (int ind = 0; ind < n_inputs ; ind++)
    {
      inputs_ptrs[ind] += add_val;
    }
    return *this;
  };

  // __host__ __device__ void calc_Gt_Gw_C_and_add(Vec<TP, 8> &Gt, Vec<TP, 8> &Gw, Mat_Sym<TP, 8> &C) const {

  //   Vec<TP, 8> G;
  //   G(0) = gx_chosen_values() * Jx();
  //   G(1) = gy_chosen_values() * Jx();
  //   G(2) = -gx_chosen_values() * Jxx_prime() - gy_chosen_values() * Jxy_prime();
  //   G(3) = gx_chosen_values() * Jy();
  //   G(4) = gy_chosen_values() * Jy();
  //   G(5) = -gx_chosen_values() * Jyx_prime() - gy_chosen_values() * Jyy_prime();
  //   G(6) = gx_chosen_values();
  //   G(7) = gy_chosen_values();

  //   Gt += G * current_level_reference_tensor_zero_mean();
  //   Gw += G * current_level_input_tensor_warped();
  //   C += G.Outer();
  // return thrust::make_tuple(Gt, Gw, C);
  // }
};

template <typename TP = float>
struct ecc_outputs_ptr
{

  Vec<TP, 8> *__restrict__ G;
  Vec<TP, 8> *__restrict__ Gt;
  Vec<TP, 8> *__restrict__ Gw;
  Mat<TP, 8> *__restrict__ C;

  __host__ __device__ ecc_outputs_ptr(Vec<TP, 8> *G_in,
                                      Vec<TP, 8> *Gt_in,
                                      Vec<TP, 8> *Gw_in,
                                      Mat<TP, 8> *C_in) : G(G_in), Gt(Gt_in), Gw(Gw_in), C(C_in){};

  // __host__ __device__ ecc_outputs_ptr<TP> &operator++()
  // {

  //   Gt++;
  //   Gw++;
  //   C++;
  //   return *this;
  // };

  // template <typename TP2>
  // __host__ __device__ ecc_outputs_ptr<TP> &operator+=(TP2 val)
  // {

  //   Gt += val;
  //   Gw += val;
  //   C += val;
  //   return *this;
  // };
};

template <typename TP>
__inline__ __device__ TP warpReduceSum(TP val)
{
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

template <typename TP>
__inline__ __device__ TP blockReduceSum(TP val)
{

  static __shared__ TP shared[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val); // Each warp performs partial reduction

  if (lane == 0)
    shared[wid] = val; // Write reduced value to shared memory

  __syncthreads(); // Wait for all partial reductions

  // read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  if (wid == 0)
    val = warpReduceSum(val); // Final reduce within first warp

  return val;
}

template <typename TP>
__global__ void ecc_reduction_ker(ecc_inputs_ptr<TP> inputs_ptr,
                                  const TP *__restrict__ current_level_reference_tensor_zero_mean_ptr,
                                  ecc_outputs_ptr<TP> outputs_ptr,
                                  const std::array<int64_t, 2> input_size)
{

  static __shared__ Mat<TP, 8> C_shared;
  static __shared__ Vec<TP, 8> Gt_shared;
  static __shared__ Vec<TP, 8> Gw_shared;

  Vec<TP, 8> Gt(0);
  Vec<TP, 8> Gw(0);
  Mat_Sym<TP, 8> C(0);

  // inputs_ptr += blockIdx.x * input_size[1] + threadIdx.x; // set the input pointer to the start of the current T

  Vec<TP, 8> G;
  float4 *G_ptr4 = (float4*)(&G);

  inputs_ptr += blockIdx.x * input_size[1] + threadIdx.x;
  Vec<TP, 8>* G_glob_ptr = &outputs_ptr.G[blockIdx.x * input_size[1] + threadIdx.x];
  current_level_reference_tensor_zero_mean_ptr+= threadIdx.x;

  for (auto i = threadIdx.x; i < input_size[1]; i += blockDim.x)
  {
    G(0) = inputs_ptr.gx_chosen_values() * inputs_ptr.Jx();
    G(1) = inputs_ptr.gy_chosen_values() * inputs_ptr.Jx();
    G(2) = -inputs_ptr.gx_chosen_values() * inputs_ptr.Jxx_prime() - inputs_ptr.gy_chosen_values() * inputs_ptr.Jxy_prime();
    G(3) = inputs_ptr.gx_chosen_values() * inputs_ptr.Jy();
    G(4) = inputs_ptr.gy_chosen_values() * inputs_ptr.Jy();
    G(5) = -inputs_ptr.gx_chosen_values() * inputs_ptr.Jyx_prime() - inputs_ptr.gy_chosen_values() * inputs_ptr.Jyy_prime();
    G(6) = inputs_ptr.gx_chosen_values();
    G(7) = inputs_ptr.gy_chosen_values();

    Gt += G * current_level_reference_tensor_zero_mean_ptr[0];
    Gw += G * inputs_ptr.current_level_input_tensor_warped();

    float4 * G_glob_ptr4 = (float4 *)G_glob_ptr;
    G_glob_ptr4[0] = G_ptr4[0];
    G_glob_ptr4[1] = G_ptr4[1];

    C += G.Outer();

    inputs_ptr += blockDim.x;
    current_level_reference_tensor_zero_mean_ptr += blockDim.x;
    G_glob_ptr += blockDim.x;
  }

#pragma unroll
  for (int ind = 0; ind < 8; ind++)
  {
    Gt_shared[ind] = blockReduceSum(Gt[ind]);
  }

#pragma unroll
  for (int ind = 0; ind < 8; ind++)
  {
    Gw_shared[ind] = blockReduceSum(Gw[ind]);
  }

#pragma unroll
  for (int i = 0; i < 8; i++)
  {
    for (int j = i; j < 8; j++)
    {
      TP C_ij = blockReduceSum(C(i, j));

      C_shared(i, j) = C_ij;

      if (j != i)
        C_shared(j, i) = C_ij;
    }
  }

  // if (threadIdx.x==0){
  //    printf("%f %f %f\n",Gt_shared(0),Gw_shared(0),C_shared(0,0));
  //  }

  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  // if(threadIdx.x==0)
  //   outputs_ptr.C[0]=C_shared;

  if ((wid == 0) and (lane < (8 * 8 / 4)))
  {

    float4 *C_shared_ptr4 = (float4 *)(&C_shared[0]) + lane;
    Mat<TP, 8> *C_glob_ptr = &outputs_ptr.C[(blockIdx.x)];
    float4 *C_glob_ptr4 = (float4 *)(C_glob_ptr) + lane;
    *C_glob_ptr4 = *C_shared_ptr4;
  }

  else if ((wid == 1) and (lane < (8 / 4)))
  {

    float4 *Gt_shared_ptr4 = (float4 *)(&Gt_shared[0]) + lane;
    Vec<TP, 8> *Gt_glob_ptr = &outputs_ptr.Gt[(blockIdx.x)];
    float4 *Gt_glob_ptr4 = (float4 *)(Gt_glob_ptr) + lane;
    *Gt_glob_ptr4 = *Gt_shared_ptr4;
  }

  else if ((wid == 2) and (lane < (8 / 4)))
  {

    float4 *Gw_shared_ptr4 = (float4 *)(&Gw_shared[0]) + lane;
    Vec<TP, 8> *Gw_glob_ptr = &outputs_ptr.Gw[(blockIdx.x)];
    float4 *Gw_glob_ptr4 = (float4 *)(Gw_glob_ptr) + lane;
    *Gw_glob_ptr4 = *Gw_shared_ptr4;
  }
}
////////////

// template <typename TP>
// void ecc_reduction_impl(const std::array<const TP *__restrict__, 10> inputs,
//                         const auto outputs,
//                         const std::array<int64_t, 2> input_size) {

//   ecc_reduction_ker<<<input_size[0], 128>>>(ecc_inputs_ptr(inputs), outputs, input_size);
// };

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