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
#include <algorithm>

// this structrue is used to store the sizes of the tensors
struct Inputs_size {
  int64_t T; // number of time points
  int64_t H; // image height
  int64_t W; // image width
  int64_t N; // number of chosen points per time point

  Inputs_size(size_t T_in, size_t H_in, size_t W_in, size_t N_in) : T(T_in), H(H_in), W(W_in), N(N_in){};
};

template <typename TP = float>
struct Inputs_ptr {

  using Raw_ptr_TP = const TP *__restrict__;
  using Mat3_ptr_TP = const Mat3<TP> *__restrict__;

  Raw_ptr_TP input_image;
  Raw_ptr_TP vx;
  Raw_ptr_TP vy;
  Raw_ptr_TP H_matrix;
  Raw_ptr_TP x_chosen_values;
  Raw_ptr_TP y_chosen_values;

  __host__ __device__ inline Inputs_ptr(const TP *input_image_in,
                                        const TP *vx_in,
                                        const TP *vy_in,
                                        const TP *H_matrix_in,
                                        const TP *x_chosen_values_in,
                                        const TP *y_chosen_values_in) : input_image(Raw_ptr_TP(input_image_in)),
                                                                        vx(Raw_ptr_TP(vx_in)),
                                                                        vy(Raw_ptr_TP(vy_in)),
                                                                        H_matrix((H_matrix_in)),
                                                                        x_chosen_values(Raw_ptr_TP(x_chosen_values_in)),
                                                                        y_chosen_values(Raw_ptr_TP(y_chosen_values_in)){};

  // this fucntion increment the chosen values
  template <typename T>
  __host__ __device__ inline Inputs_ptr<TP> operator+=(T shft) {
    // int64_t shft = page_num * page_size;
    x_chosen_values += shft;
    y_chosen_values += shft;
    return *this;
  };

  // this fucntion set the input pointers to a given page (T) within the tensor
  __host__ __device__ inline Inputs_ptr<TP> set_page(int page_num, Inputs_size sizes) {
    // int64_t shft = page_num * page_size;
    size_t im_size = sizes.H * sizes.W;
    input_image += page_num * im_size;
    vx += page_num * im_size;
    vy += page_num * im_size;
    H_matrix += page_num*9;
    x_chosen_values += page_num * sizes.N;
    y_chosen_values += page_num * sizes.N;

    return *this;
  };

  // this function is used to set the position of the input pointers to a given time (T) and position (n) within.
  //  note here that since H is the same for all points with a given T, we only increment the H_matrix by the page number

  __host__ __device__ inline Inputs_ptr<TP> set_pos(int page_num, int page_pos, Inputs_size sizes) {
    size_t im_size = sizes.H * sizes.W;
    input_image += page_num * im_size;
    vx += page_num * im_size;
    vy += page_num * im_size;
    H_matrix += page_num*9;
    x_chosen_values += page_num * sizes.N + page_pos;
    y_chosen_values += page_num * sizes.N + page_pos;
    return *this;
  };


};



template <typename TP = float>
struct Inputs_ptr_no_grad {

  using Raw_ptr_TP = const TP *__restrict__;
  using Mat3_ptr_TP = const Mat3<TP> *__restrict__;

  Raw_ptr_TP input_image;
  Raw_ptr_TP H_matrix;
  Raw_ptr_TP x_chosen_values;
  Raw_ptr_TP y_chosen_values;

  __host__ __device__ inline Inputs_ptr_no_grad(const TP *input_image_in,
                                        const TP *H_matrix_in,
                                        const TP *x_chosen_values_in,
                                        const TP *y_chosen_values_in) : input_image(Raw_ptr_TP(input_image_in)),
                                                                        H_matrix((H_matrix_in)),
                                                                        x_chosen_values(Raw_ptr_TP(x_chosen_values_in)),
                                                                        y_chosen_values(Raw_ptr_TP(y_chosen_values_in)){};

  // this fucntion increment the chosen values
  template <typename T>
  __host__ __device__ inline Inputs_ptr_no_grad<TP> operator+=(T shft) {
    // int64_t shft = page_num * page_size;
    x_chosen_values += shft;
    y_chosen_values += shft;
    return *this;
  };

  // this fucntion set the input pointers to a given page (T) within the tensor
  __host__ __device__ inline Inputs_ptr_no_grad<TP> set_page(int page_num, Inputs_size sizes) {
    // int64_t shft = page_num * page_size;
    size_t im_size = sizes.H * sizes.W;
    input_image += page_num * im_size;
    H_matrix += page_num*9;
    x_chosen_values += page_num * sizes.N;
    y_chosen_values += page_num * sizes.N;

    return *this;
  };

  // this function is used to set the position of the input pointers to a given time (T) and position (n) within.
  //  note here that since H is the same for all points with a given T, we only increment the H_matrix by the page number

  __host__ __device__ inline Inputs_ptr_no_grad<TP> set_pos(int page_num, int page_pos, Inputs_size sizes) {
    size_t im_size = sizes.H * sizes.W;
    input_image += page_num * im_size;
    H_matrix += page_num*9;
    x_chosen_values += page_num * sizes.N + page_pos;
    y_chosen_values += page_num * sizes.N + page_pos;
    return *this;
  };


};


// this structrue is used to store the outputs of the ECC algorithm, nothing more than the G, Gt, Gw and C matrices
// it is only used to make the code more readable
template <typename TP = float>
struct Outputs_ptr {

  using Raw_ptr_TP = TP *__restrict__;

  Raw_ptr_TP input_image_warped;
  Raw_ptr_TP vx_warped;
  Raw_ptr_TP vy_warped;

  __host__ __device__ inline Outputs_ptr(TP *input_image_warped_in,
                                         TP *vx_warped_in,
                                         TP *vy_warped_in) : input_image_warped(input_image_warped_in),
                                                             vx_warped(vx_warped_in),
                                                             vy_warped(vy_warped_in){};

  // set the output pointers to a output pos within the tensors
  __host__ __device__ inline Outputs_ptr<TP> set_pos(int page_num, int page_pos, Inputs_size sizes) {
    input_image_warped += page_num * sizes.N + page_pos;
    vx_warped += page_num * sizes.N + page_pos;
    vy_warped += page_num * sizes.N + page_pos;
    return *this;
  };

  template <typename T>
  __host__ __device__ inline Outputs_ptr<TP> operator+=(T shft) {
    input_image_warped += shft;
    vx_warped += shft;
    vy_warped += shft;
    return *this;
  };
};

// this kernel is used to compute the bileaner interpolation of the input image and vx and vy
template <typename TP = float>
__global__ void bilinear_interpolation_ker(Inputs_ptr<TP> inputs_ptr,
                                           Outputs_ptr<TP> outputs_ptr,
                                           const Inputs_size inputs_size) {

  // set the position of the input pointers to the current block and current thread within the block
  inputs_ptr.set_pos(blockIdx.x, threadIdx.x, inputs_size); //
  outputs_ptr.set_pos(blockIdx.x, threadIdx.x, inputs_size);

  Mat3<TP> H_matrix;
  H_matrix(2, 2) = TP(1);
  // get the lane id
  int lane_id = threadIdx.x % warpSize;
  // we will load the H matrix using a suffle// note that since H_matrix(2,2) is always 1, we can ignore it and load only the first 8 elements
  TP H_matrix_elem;
  if (lane_id < 8) {
    // load a single element of the H matrix
    H_matrix_elem = inputs_ptr.H_matrix[lane_id];
  }

  // now shuffle the elements of the H matrix to the correct position
  for (int i = 0; i < 8; i++) {
    H_matrix[i] = __shfl_sync(0xffffffff, H_matrix_elem, i);
  }

  // we now loop over the chosen values and compute the transformed coordinates with a stride of one block
  // load chosen_values
  for (int ind = threadIdx.x; ind < inputs_size.N; ind+= blockDim.x) {
    Vec2<TP> chosen_values(inputs_ptr.x_chosen_values[0], inputs_ptr.y_chosen_values[0]);

    TP z_ratio = H_matrix(2, 0) * chosen_values(0) + H_matrix(2, 1) * chosen_values(1) + H_matrix(2, 2);

    // calulate the transformed coordinates
    // Vec2<TP> chosen_values_corrected(TP(2) * (H_matrix(0, 0) * chosen_values(0) + H_matrix(0, 1) * chosen_values(1) + H_matrix(0, 2)) / denom / inputs_size.W - TP(1),
    //                                      TP(2) * (H_matrix(1, 0) * chosen_values(0) + H_matrix(1, 1) * chosen_values(1) + H_matrix(1, 2)) / denom / inputs_size.H - TP(1));

    Vec2<TP> chosen_values_corrected( (H_matrix(0, 0) * chosen_values(0) + H_matrix(0, 1) * chosen_values(1) + H_matrix(0, 2)) / z_ratio,
                                      (H_matrix(1, 0) * chosen_values(0) + H_matrix(1, 1) * chosen_values(1) + H_matrix(1, 2)) / z_ratio);


    // calculate the bilinear interpolation
    // fisrt calculate the 4 points around the transformed coordinates

    Vec2<int> p0_0(floor(chosen_values_corrected(0)), floor(chosen_values_corrected(1)));

    // make sure the points are within the image
    p0_0(0) = max(0, min(p0_0(0), int(inputs_size.W - 2)));
    p0_0(1) = max(0, min(p0_0(1), int(inputs_size.H - 2)));



    //printf("p0_0: %d, %d \n", p0_0(0), p0_0(1));
    // caclute the interpolation postions for the 4 points using the distance from the transformed coordinates and the formula
    //  f(x,y)=(1-x ,x)*(f(0,0) , f(0,1) , f(1,0) , f(1,1))* (1-y ;y)
    Vec2<TP> x_pos;
    x_pos(1) = chosen_values_corrected(0) - p0_0(0);
    x_pos(0) = TP(1) - x_pos(1);

    Vec2<TP> y_pos;
    y_pos(1) = chosen_values_corrected(1) - p0_0(1);
    y_pos(0) = TP(1) - y_pos(1);

    // we will caclute the other points later by adding 1 to the x and y coordinates

    // we will store the 4 pointes in a 2*2 matrix
    Mat2<TP> vals;
    vals(0, 0) = inputs_ptr.input_image[p0_0(0) + p0_0(1) * inputs_size.W];
    vals(0, 1) = inputs_ptr.input_image[p0_0(0) + (p0_0(1) + 1) * inputs_size.W];
    vals(1, 0) = inputs_ptr.input_image[(p0_0(0) + 1) + p0_0(1) * inputs_size.W];
    vals(1, 1) = inputs_ptr.input_image[(p0_0(0) + 1) + (p0_0(1) + 1) * inputs_size.W];

    // now we can calculate the bilinear interpolation and store it in the output
  
    outputs_ptr.input_image_warped[0] = x_pos * vals * y_pos;

    // now do the same for vx and vy
    vals(0, 0) = inputs_ptr.vx[p0_0(0) + p0_0(1) * inputs_size.W];
    vals(0, 1) = inputs_ptr.vx[p0_0(0) + (p0_0(1) + 1) * inputs_size.W];
    vals(1, 0) = inputs_ptr.vx[(p0_0(0) + 1) + p0_0(1) * inputs_size.W];
    vals(1, 1) = inputs_ptr.vx[(p0_0(0) + 1) + (p0_0(1) + 1) * inputs_size.W];

    outputs_ptr.vx_warped[0] = x_pos * vals * y_pos;

    // vy
    vals(0, 0) = inputs_ptr.vy[p0_0(0) + p0_0(1) * inputs_size.W];
    vals(0, 1) = inputs_ptr.vy[p0_0(0) + (p0_0(1) + 1) * inputs_size.W];
    vals(1, 0) = inputs_ptr.vy[(p0_0(0) + 1) + p0_0(1) * inputs_size.W];
    vals(1, 1) = inputs_ptr.vy[(p0_0(0) + 1) + (p0_0(1) + 1) * inputs_size.W];

    outputs_ptr.vy_warped[0] = x_pos * vals * y_pos;

    // now we can move to the next chosen value
    inputs_ptr += blockDim.x;
    outputs_ptr += blockDim.x;
  }
}
////////////

// this kernel is used to compute the bileaner interpolation of the input image and vx and vy
template <typename TP = float>
__global__ void bilinear_interpolation_no_grad_ker(Inputs_ptr_no_grad<TP> inputs_ptr,
                                           Outputs_ptr<TP> outputs_ptr,
                                           const Inputs_size inputs_size) {

  // set the position of the input pointers to the current block and current thread within the block
  inputs_ptr.set_pos(blockIdx.x, threadIdx.x, inputs_size); //
  outputs_ptr.set_pos(blockIdx.x, threadIdx.x, inputs_size);

  Mat3<TP> H_matrix;
  H_matrix(2, 2) = TP(1);
  // get the lane id
  int lane_id = threadIdx.x % warpSize;
  // we will load the H matrix using a suffle// note that since H_matrix(2,2) is always 1, we can ignore it and load only the first 8 elements
  TP H_matrix_elem;
  if (lane_id < 8) {
    // load a single element of the H matrix
    H_matrix_elem = inputs_ptr.H_matrix[lane_id];
  }

  // now shuffle the elements of the H matrix to the correct position
  for (int i = 0; i < 8; i++) {
    H_matrix[i] = __shfl_sync(0xffffffff, H_matrix_elem, i);
  }

  // we now loop over the chosen values and compute the transformed coordinates with a stride of one block
  // load chosen_values
  for (int ind = threadIdx.x; ind < inputs_size.N; ind+= blockDim.x) {
    Vec2<TP> chosen_values(inputs_ptr.x_chosen_values[0], inputs_ptr.y_chosen_values[0]);

    TP z_ratio = H_matrix(2, 0) * chosen_values(0) + H_matrix(2, 1) * chosen_values(1) + H_matrix(2, 2);

    // calulate the transformed coordinates
    // Vec2<TP> chosen_values_corrected(TP(2) * (H_matrix(0, 0) * chosen_values(0) + H_matrix(0, 1) * chosen_values(1) + H_matrix(0, 2)) / denom / inputs_size.W - TP(1),
    //                                      TP(2) * (H_matrix(1, 0) * chosen_values(0) + H_matrix(1, 1) * chosen_values(1) + H_matrix(1, 2)) / denom / inputs_size.H - TP(1));

    Vec2<TP> chosen_values_corrected( (H_matrix(0, 0) * chosen_values(0) + H_matrix(0, 1) * chosen_values(1) + H_matrix(0, 2)) / z_ratio,
                                      (H_matrix(1, 0) * chosen_values(0) + H_matrix(1, 1) * chosen_values(1) + H_matrix(1, 2)) / z_ratio);

    // calculate the bilinear interpolation
    // fisrt calculate the 4 points around the transformed coordinates

    Vec2<int> p0_0(floor(chosen_values_corrected(0)), floor(chosen_values_corrected(1)));

    // make sure that the points are within the image
    p0_0(0) = max(0, min(p0_0(0), int(inputs_size.W - 2)));
    p0_0(1) = max(0, min(p0_0(1), int(inputs_size.H - 2)));

    //printf("p0_0: %d, %d \n", p0_0(0), p0_0(1));
    // caclute the interpolation postions for the 4 points using the distance from the transformed coordinates and the formula
    //  f(x,y)=(1-x ,x)*(f(0,0) , f(0,1) , f(1,0) , f(1,1))* (1-y ;y)
    Vec2<TP> x_pos;
    x_pos(1) = chosen_values_corrected(0) - p0_0(0);
    x_pos(0) = TP(1) - x_pos(1);

    Vec2<TP> y_pos;
    y_pos(1) = chosen_values_corrected(1) - p0_0(1);
    y_pos(0) = TP(1) - y_pos(1);

    // we will caclute the other points later by adding 1 to the x and y coordinates

    // we will store the 4 pointes in a 2*2 matrix
    Mat2<TP> vals;
    vals(0, 0) = inputs_ptr.input_image[p0_0(0) + p0_0(1) * inputs_size.W];
    vals(0, 1) = inputs_ptr.input_image[p0_0(0) + (p0_0(1) + 1) * inputs_size.W];
    vals(1, 0) = inputs_ptr.input_image[(p0_0(0) + 1) + p0_0(1) * inputs_size.W];
    vals(1, 1) = inputs_ptr.input_image[(p0_0(0) + 1) + (p0_0(1) + 1) * inputs_size.W];

    // now we can calculate the bilinear interpolation and store it in the output

    outputs_ptr.input_image_warped[0] = x_pos * vals * y_pos;

    // now to the gradients
    
    // the x gradinet is calcuted by the formula f'(x,y)=(1-y ,y)*(f(0,0) , f(0,1) , f(1,0) , f(1,1))* (-1 ;1)
    // this vetor holds the x gradient at y=0 and y=1
    Vec2<TP> gx( -vals(0, 0) + vals(0, 1), -vals(1, 0) + vals(1, 1));
    // to get the x gradient at y, we need to multiply the x gradient at y=0 and y=1 by the y position i.e gx(x,y)= (1-y)*gx(y=0)+y*gx(y=1)

    outputs_ptr.vx_warped[0] =  y_pos*gx;

    // the y gradinet is calcuted by the formula f'(x,y)= (-1,1)(f(0,0) , f(0,1) , f(1,0) , f(1,1))* (1-x ;x)
    // this vetor holds the y gradient at x=0 and x=1
    Vec2<TP> gy( -vals(0, 0) + vals(1, 0), -vals(0, 1) + vals(1, 1));
    
    // to get the x gradient at x, we need to multiply the u gradient at x=0 and x=1 by the x position i.e gx(x,y)= (1-x)*gy(x=0)+x*gy(x=1)
    outputs_ptr.vy_warped[0] = x_pos*gy;

    // now we can move to the next chosen value
    inputs_ptr += blockDim.x;
    outputs_ptr += blockDim.x;
  }
}
////////////