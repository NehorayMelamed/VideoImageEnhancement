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

// make a cholesky decomposition
// using a single warp

// this structrue is used to store the inputs to the ECC algorithm
template <typename TP = float>
struct ECC_inputs_ptr
{

  using Raw_ptr_TP = const TP *__restrict__;
  using Mat3_ptr_TP = const Mat3<TP> *__restrict__;

  Mat3_ptr_TP H_matrix;
  Raw_ptr_TP current_level_reference_tensor_zero_mean;
  Raw_ptr_TP current_level_input_tensor_warped;
  Raw_ptr_TP Jx_chosen_values;
  Raw_ptr_TP Jy_chosen_values;
  Raw_ptr_TP gx_chosen_values;
  Raw_ptr_TP gy_chosen_values;

  ECC_inputs_ptr(const TP *H_matrix_in,
                 const TP *current_level_reference_tensor_zero_mean_in,
                 const TP *current_level_input_tensor_warped_in,
                 const TP *Jx_chosen_values_in,
                 const TP *Jy_chosen_values_in,
                 const TP *gx_chosen_values_in,
                 const TP *gy_chosen_values_in) : H_matrix(Mat3_ptr_TP(H_matrix_in)),
                                                  current_level_reference_tensor_zero_mean(Raw_ptr_TP(current_level_reference_tensor_zero_mean_in)),
                                                  current_level_input_tensor_warped(Raw_ptr_TP(current_level_input_tensor_warped_in)),
                                                  Jx_chosen_values(Raw_ptr_TP(Jx_chosen_values_in)),
                                                  Jy_chosen_values(Raw_ptr_TP(Jy_chosen_values_in)),
                                                  gx_chosen_values(Raw_ptr_TP(gx_chosen_values_in)),
                                                  gy_chosen_values(Raw_ptr_TP(gy_chosen_values_in)){};

  // this command is used to increment the pointers. note that the H_matrix is not incremented as it is the same for all points with a givven T
  __host__ __device__ ECC_inputs_ptr<TP> &operator++()
  {
    // H_matrix++;
    current_level_reference_tensor_zero_mean++;
    current_level_input_tensor_warped++;
    Jx_chosen_values++;
    Jy_chosen_values++;
    gx_chosen_values++;
    gy_chosen_values++;
    return *this;
  };

  template <typename TP2>
  __host__ __device__ ECC_inputs_ptr<TP> &operator+=(TP2 add_val)
  {
    // H_matrix += add_val;
    current_level_reference_tensor_zero_mean += add_val;
    current_level_input_tensor_warped += add_val;
    Jx_chosen_values += add_val;
    Jy_chosen_values += add_val;
    gx_chosen_values += add_val;
    gy_chosen_values += add_val;
    return *this;
  };

  __host__ __device__ ECC_inputs_ptr<TP> set_page(int page_num, int64_t page_size)
  {
    int64_t shft = page_num * page_size;
    H_matrix += page_num;
    current_level_input_tensor_warped += shft;
    Jx_chosen_values += shft;
    Jy_chosen_values += shft;
    gx_chosen_values += shft;
    gy_chosen_values += shft;
    return *this;
  };

  // this function is used to set the position of the input pointers to a given time (T) and position (n) within.
  //  note here that since H is the same for all points with a given T, we only increment the H_matrix by the page number
  //  while  current_level_reference_tensor_zero_mean is the same for all points with a given T, we only increment the current_level_reference_tensor_zero_mean by the page position n
  __host__ __device__ ECC_inputs_ptr<TP> set_pos(int page_num, int page_pos, int64_t page_size)
  {
    int64_t shft = page_num * page_size + page_pos;
    H_matrix += page_num;
    current_level_input_tensor_warped += shft;
    Jx_chosen_values += shft;
    Jy_chosen_values += shft;
    gx_chosen_values += shft;
    gy_chosen_values += shft;
    current_level_reference_tensor_zero_mean += page_pos;
    return *this;
  };
};

// this structrue is used to store the outputs of the ECC algorithm, nothing more than the G, Gt, Gw and C matrices
// it is only used to make the code more readable
template <typename TP = float>
struct ECC_outputs_ptr
{

  Vec<TP, 8> *__restrict__ G;
  TP *__restrict__ CLITW_norm2;
  TP *__restrict__ CLITW_dot_CLRTZM;
  Vec<TP, 8> *__restrict__ Gt;
  Vec<TP, 8> *__restrict__ Gw;
  Mat<TP, 8> *__restrict__ C;

  __host__ __device__ ECC_outputs_ptr(Vec<TP, 8> *G_in,
                                      TP *CLITW_norm2_in,
                                      TP *CLITW_dot_CLRTZM_in,
                                      Vec<TP, 8> *Gt_in,
                                      Vec<TP, 8> *Gw_in,
                                      Mat<TP, 8> *C_in) : G(G_in), CLITW_norm2(CLITW_norm2_in), CLITW_dot_CLRTZM(CLITW_dot_CLRTZM_in), Gt(Gt_in), Gw(Gw_in), C(C_in){};

  __host__ __device__ ECC_outputs_ptr<TP> set_pos(int page_num, int page_pos, int64_t page_size)
  {
    int64_t shft = page_num * page_size + page_pos;
    G += shft;
    Gw += page_num;
    Gt += page_num;
    C += page_num;
    CLITW_norm2 += page_num;
    CLITW_dot_CLRTZM += page_num;
    return *this;
  };
  // __host__ __device__ Ecc_outputs_ptr<TP> &operator++() {

  //   Gt++;
  //   Gw++;
  //   C++;
  //   return *this;
  // };

  // template <typename TP2>
  // __host__ __device__ Ecc_outputs_ptr<TP> &operator+=(TP2 val) {

  //   Gt += val;
  //   Gw += val;
  //   C += val;
  //   return *this;
  // };
};

// this function makes a warp reduction of the input value
template <typename TP>
__inline__ __device__ TP warpReduceSum(TP val)
{
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

// this function makes a block wide reduction of the input value
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

// this function is used to compute the G, Gt, Gw and C matrices
template <typename TP = float>
__global__ void ecc_reduction_ker(ECC_inputs_ptr<TP> inputs_ptr,
                                  ECC_outputs_ptr<TP> outputs_ptr,
                                  const std::array<int64_t, 2> input_size)
{

  // set the position of the input pointers to the current block and current thread within the block
  inputs_ptr.set_pos(blockIdx.x, threadIdx.x, input_size[1]); //
  outputs_ptr.set_pos(blockIdx.x, threadIdx.x, input_size[1]);

  // allocate local memory elements
  Vec<TP, 8> Gt(0);        // this is the Gt vector stored in the thread local memory
  Vec<TP, 8> Gw(0);        // this is the Gw vector stored in the thread local memory
  Mat_Sym<TP, 8> C(0);     // this is the C matrix stored in the thread local memory
  Vec<TP, 8> G;            // this is the G vector stored in the thread local memory
  TP CLITW_norm = 0;       // this is the norm of current_level_input_tensor_warped stored in the thread local memory
  TP CLITW_dot_CLRTZM = 0; // this is current_level_input_tensor_warped * current_level_reference_tensor_zero_mean stored in the thread local memory

  // alias the pointer to G to a float4 pointer so we can use vectorized write
  float4 *G_ptr4 = (float4 *)(&G);
  Vec<TP, 8> *G_glob_ptr = outputs_ptr.G;
  // Vec<TP, 8> *G_glob_ptr = outputs_ptr.G + blockIdx.x * input_size[1] + threadIdx.x;

  // allocate shared memory elements
  static __shared__ Mat<TP, 8> C_shared;       // this is the shared memory version of the C matrix
  static __shared__ Vec<TP, 8> Gt_shared;      // this is the shared memory version of the Gt vector
  static __shared__ Vec<TP, 8> Gw_shared;      // this is the shared memory version of the Gw vector

  // load the H matrix into local memory so we can use it in the loop
  Mat3<TP> H_matrix;
  H_matrix(2, 2) = 1; // the last element of the H matrix is always 1
  int lane = threadIdx.x % warpSize;
  TP H_matrix_elem; // this will store a single element of the H matrix to be shuffled to the correct position
  if (lane < 8)
  {                                                            // we cant use vectorzied load here because the H matrix is not aligned so insted we shuffle within the warp
    H_matrix_elem = ((const TP *)(inputs_ptr.H_matrix))[lane]; // hence each thread in warp loads one element, as long as the laneid is less than 8
  }

  for (auto i = 0; i < 8; i++)
  {
    H_matrix[i] = __shfl_sync(0xffffffff, H_matrix_elem, i); // now shuffle the elements to the correct position
  }

  // if ((threadIdx.x == 0) and (blockIdx.x == 1))
  //   H_matrix.print();
  // loop over the input data, each warp it responsible for a single T. each thread loops over the n in that T,
  // thus each thread makes jumps of blockDim.x in the n direction
  for (auto i = threadIdx.x; i < input_size[1]; i += blockDim.x)
  {

    // load the J values into local memory
    Vec2<TP> J_chosen_values = {inputs_ptr.Jx_chosen_values[0], inputs_ptr.Jy_chosen_values[0]};

    // compute the inverse of the denominator
    TP denom_inverse = TP(1) / (H_matrix(2, 0) * J_chosen_values(0) + H_matrix(2, 1) * J_chosen_values(1) + TP(1));

    // compute the xy_prime_reshaped vector
    Vec2<TP> xy_prime_reshaped = {(H_matrix(0, 0) * J_chosen_values(0) +
                                   H_matrix(0, 1) * J_chosen_values(1) +
                                   H_matrix(0, 2)) *
                                      denom_inverse,
                                  (H_matrix(1, 0) * J_chosen_values(0) +
                                   H_matrix(1, 1) * J_chosen_values(1) +
                                   H_matrix(1, 2)) *
                                      denom_inverse};

    // update J_chosen_values
    J_chosen_values *= denom_inverse;

    // compute the J_prime matrix
    Mat2<TP> J_prime = J_chosen_values.Outer(xy_prime_reshaped);

    // load the g values into local memory
    Vec2<TP> g_chosen_values = {inputs_ptr.gx_chosen_values[0], inputs_ptr.gy_chosen_values[0]};

    // compute the G vector
    G(0) = g_chosen_values(0) * J_chosen_values(0);
    G(1) = g_chosen_values(1) * J_chosen_values(0);
    G(2) = -g_chosen_values(0) * J_prime(0, 0) - g_chosen_values(1) * J_prime(0, 1);
    G(3) = g_chosen_values(0) * J_chosen_values(1);
    G(4) = g_chosen_values(1) * J_chosen_values(1);
    G(5) = -g_chosen_values(0) * J_prime(1, 0) - g_chosen_values(1) * J_prime(1, 1);
    G(6) = g_chosen_values(0);
    G(7) = g_chosen_values(1);

    // calculate the Gt and Gw vectors
    TP current_level_reference_tensor_zero_mean = inputs_ptr.current_level_reference_tensor_zero_mean[0];
    TP current_level_input_tensor_warped = inputs_ptr.current_level_input_tensor_warped[0];

    Gt += G * current_level_reference_tensor_zero_mean;
    Gw += G * current_level_input_tensor_warped;

    // add the current_level_input_tensor_warped[0] to the clatw_norm
    CLITW_norm += current_level_input_tensor_warped * current_level_input_tensor_warped;

    // add the current_level_reference_tensor_zero_warped *current_level_reference_tensor_zero_mean to the clatw_dot_clatw
    CLITW_dot_CLRTZM += current_level_input_tensor_warped * current_level_reference_tensor_zero_mean;

    // write the G vector to global memory using vectorized write
    float4 *G_glob_ptr4 = (float4 *)G_glob_ptr;
    G_glob_ptr4[0] = G_ptr4[0];
    G_glob_ptr4[1] = G_ptr4[1];

    // calculate the C matrix
    C += G.Outer();

    // increment the pointers to the next element
    inputs_ptr += blockDim.x;
    G_glob_ptr += blockDim.x;
    // current_level_reference_tensor_zero_mean_ptr += blockDim.x;
  }

  // at this point we have the clatw_norm, clatw_dot_clatw ,Gt, Gw and C matrix for this block, we need to reduce them to a single value
  // we do this by using a blockReduceSum.

  // reduce the clatw_norm and clatw_dot_clatw write the result to the output
  outputs_ptr.CLITW_norm2[0] = blockReduceSum(CLITW_norm);
  outputs_ptr.CLITW_dot_CLRTZM[0] = blockReduceSum(CLITW_dot_CLRTZM);

// reduce the Gt and Gw vectors and the C matrix
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
  //////////////////////////////////////////////////////////

  // we need to write the Gt and Gw vectors and the C matrix to global memory, where warp 0 writes C, war[ 1 writes Gt and warp 2 writes Gw
  int wid = threadIdx.x / warpSize;

  // if(threadIdx.x==0)
  //   outputs_ptr.C[0]=C_shared;

  if ((wid == 0) and (lane < (8 * 8 / 4)))
  {
    // alias the shared memory to float4
    float4 *C_shared_ptr4 = (float4 *)(&C_shared[0]) + lane;

    // Mat<TP, 8> *C_glob_ptr = outputs_ptr.C + blockIdx.x;
    Mat<TP, 8> *C_glob_ptr = outputs_ptr.C;
    float4 *C_glob_ptr4 = (float4 *)(C_glob_ptr) + lane;
    *C_glob_ptr4 = *C_shared_ptr4;
  }

  else if ((wid == 1) and (lane < (8 / 4)))
  {

    float4 *Gt_shared_ptr4 = (float4 *)(&Gt_shared[0]) + lane;
    // Vec<TP, 8> *Gt_glob_ptr = outputs_ptr.Gt + blockIdx.x;
    Vec<TP, 8> *Gt_glob_ptr = outputs_ptr.Gt;
    float4 *Gt_glob_ptr4 = (float4 *)(Gt_glob_ptr) + lane;
    *Gt_glob_ptr4 = *Gt_shared_ptr4;
  }

  else if ((wid == 2) and (lane < (8 / 4)))
  {
    float4 *Gw_shared_ptr4 = (float4 *)(&Gw_shared[0]) + lane;
    // Vec<TP, 8> *Gw_glob_ptr = outputs_ptr.Gw + blockIdx.x;
    Vec<TP, 8> *Gw_glob_ptr = outputs_ptr.Gw;
    float4 *Gw_glob_ptr4 = (float4 *)(Gw_glob_ptr) + lane;
    *Gw_glob_ptr4 = *Gw_shared_ptr4;
  }
}
////////////
