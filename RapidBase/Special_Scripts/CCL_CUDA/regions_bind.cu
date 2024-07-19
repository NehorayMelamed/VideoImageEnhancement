#pragma once

#include <algorithm>
#include <array>
#include <bitset>
#include <chrono>
#include <climits>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <torch/torch.h>
#include <vector>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/iterator/discard_iterator.h>
#include "CCL.cu"
#include "blob_centers.cu"

// #include "conv_data_types.cu"

auto CCL(torch::Tensor outliers_tensor)
{

  std::cout << "TST DATA PTR inner" << outliers_tensor.data_ptr<bool>() << std::endl;

  auto input_sizes = outliers_tensor.sizes();
  // std::cout << input_sizes[0] << input_sizes[1] << input_sizes[2] << std::endl;
  auto output_tensor_options =
      torch::TensorOptions()
          .dtype(torch::kInt32)
          .layout(torch::kStrided)
          .device(torch::kCUDA)
          .requires_grad(false);

  torch::Tensor CCL_labels_torch = torch::empty(outliers_tensor.sizes(), output_tensor_options);

  bool *input_tensor_raw_ptr = outliers_tensor.data_ptr<bool>();

  int *CCL_tensor_raw_ptr = CCL_labels_torch.data_ptr<int>();

#pragma unroll
  for (int z = 0; z < input_sizes[0]; z++)
  {

    connectedComponentLabeling(CCL_tensor_raw_ptr, input_tensor_raw_ptr, input_sizes[2], input_sizes[1]);
    CCL_tensor_raw_ptr += input_sizes[1] * input_sizes[2];
    input_tensor_raw_ptr += input_sizes[1] * input_sizes[2];
  }
  return CCL_labels_torch;
}

auto CC_Blob_Centers(torch::Tensor &CC_labeled_image)
{

  auto input_sizes = CC_labeled_image.sizes();
  auto img_len = input_sizes[1] * input_sizes[2];
  // cout << input_sizes << endl;
  // std::cout << input_sizes[0] << input_sizes[1] << input_sizes[2] << std::endl;
  auto output_tensor_options =
      torch::TensorOptions()
          .dtype(torch::kInt32)
          .layout(torch::kStrided)
          .device(torch::kCUDA)
          .requires_grad(false);

  auto n_blobs_tensor_options =
      torch::TensorOptions()
          .dtype(torch::kInt32)
          .layout(torch::kStrided)
          .device(torch::kCPU)
          .requires_grad(false);

  thrust::device_ptr<int> CC_labeled_image_ptr = thrust::device_pointer_cast<int>(CC_labeled_image.data_ptr<int>());

  torch::Tensor blob_inds = torch::empty(img_len, output_tensor_options);
  thrust::device_ptr<int> blob_inds_ptr = thrust::device_pointer_cast<int>(blob_inds.data_ptr<int>());

  torch::Tensor blob_labels = torch::empty(img_len, output_tensor_options);
  thrust::device_ptr<int> blob_labels_ptr = thrust::device_pointer_cast<int>(blob_labels.data_ptr<int>());

  torch::Tensor n_blobs = torch::empty(input_sizes[0], n_blobs_tensor_options);
  //int *n_blobs_ptr = n_blobs.data_ptr<int>();
  // thrust::device_ptr<int> blob_labels_ptr = thrust::device_pointer_cast<int>(blob_lables.data_ptr<int>());

  torch::Tensor blob_stats = torch::empty({input_sizes[0], (img_len / 4 + 1) * 3}, output_tensor_options);
  //thrust::device_ptr<int3> blob_stats_ptr = thrust::device_pointer_cast<int3>(blob_stats.data_ptr<int3>());
  //thrust::device_ptr<Blob_stat> blob_stats_ptr = thrust::device_pointer_cast<Blob_stat>(blob_stats.data_ptr<Blob_stat>());
  int* blob_stats_raw_ptr =  blob_stats.data_ptr<int>();
  thrust::device_ptr<Blob_stat> blob_stats_ptr = thrust::device_pointer_cast<Blob_stat>((Blob_stat *) blob_stats_raw_ptr );

  for (int T = 0; T < input_sizes[0]; T++)
  {

    auto CC_input_zip_iter = thrust::make_zip_iterator(CC_labeled_image_ptr, thrust::make_counting_iterator(0));
    auto labels_zip_iter = thrust::make_zip_iterator(blob_labels_ptr, blob_inds_ptr);

    auto n_points_iter = thrust::copy_if(CC_input_zip_iter,
                                   CC_input_zip_iter + img_len,
                                   labels_zip_iter,
                                   get_non_zero_labels());


    size_t n_points = n_points_iter - labels_zip_iter;                

    thrust::sort_by_key(blob_labels_ptr, blob_labels_ptr + n_points, blob_inds_ptr);

    Make_pos_from_index make_pos_from_index_op(input_sizes[2]);
    //thrust::transform_iterator<Make_pos_from_index, thrust::device_ptr<int> > blob_inds_iter(blob_inds_ptr, make_pos_from_index_op);
    thrust::transform_iterator blob_inds_iter(blob_inds_ptr, make_pos_from_index_op);

    auto reduce_out_iter = thrust::reduce_by_key(blob_labels_ptr, blob_labels_ptr + n_points, blob_inds_iter, thrust::make_discard_iterator() , blob_stats_ptr);

    n_blobs[T] = thrust::get<1>(reduce_out_iter)-blob_stats_ptr;
    CC_labeled_image_ptr += img_len;
    blob_stats_ptr += img_len / 4 + 1;
  }

  return std::make_tuple(blob_stats, n_blobs);
}
