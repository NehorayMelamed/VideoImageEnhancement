

#include "CCL.cu"
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
#include <thrust/device_ptr.h>
#include <torch/extension.h>
#include <torch/library.h>
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
#include <thrust/iterator/transform_output_iterator.h>
#include "CCL.cu"
#include "blob_centers.cu"

using std::cout;
using std::endl;
// #include "conv_data_types.cu"

#define gpuErrchk(ans)                    \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

auto CCL(torch::Tensor &outliers_tensor)
{
  auto input_sizes = outliers_tensor.sizes();

  // cout << input_sizes << endl;
  // std::cout << input_sizes[0] << input_sizes[1] << input_sizes[2] << std::endl;
  auto output_tensor_options =
      torch::TensorOptions()
          .dtype(torch::kInt32)
          .layout(torch::kStrided)
          .device(torch::kCUDA, 0)
          .requires_grad(false);

  torch::Tensor CCL_labels_torch = torch::empty(outliers_tensor.sizes(), output_tensor_options);
  bool *input_tensor_raw_ptr = outliers_tensor.data_ptr<bool>();

  int *CCL_tensor_raw_ptr = CCL_labels_torch.data_ptr<int>();

#pragma unroll
  for (int z = 0; z < input_sizes[0]; z++)
  {
    connectedComponentLabeling(CCL_tensor_raw_ptr, input_tensor_raw_ptr, input_sizes[2], input_sizes[1]);
    // gpuErrchk(cudaDeviceSynchronize());
    // cout << z << endl;
    CCL_tensor_raw_ptr += input_sizes[1] * input_sizes[2];
    input_tensor_raw_ptr += input_sizes[1] * input_sizes[2];
  }
  return CCL_labels_torch;
}
/*
auto CC_Blob_Centers_Formatted(torch::Tensor &CC_labeled_image, float max_blob_ratio)
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



  thrust::device_ptr<int> CC_labeled_image_ptr = thrust::device_pointer_cast<int>(CC_labeled_image.data_ptr<int>());

  torch::Tensor blob_inds = torch::empty(img_len, output_tensor_options);
  thrust::device_ptr<int> blob_inds_ptr = thrust::device_pointer_cast<int>(blob_inds.data_ptr<int>());

  torch::Tensor blob_labels = torch::empty(img_len, output_tensor_options);
  thrust::device_ptr<int> blob_labels_ptr = thrust::device_pointer_cast<int>(blob_labels.data_ptr<int>());

  torch::Tensor blob_stats = torch::empty({ int64_t(int(input_sizes[0])* (img_len / 4 + 1) * max_blob_ratio) , 4}, output_tensor_options);
  int *blob_stats_raw_ptr = blob_stats.data_ptr<int>();
  thrust::device_ptr<int4> blob_stats_ptr = thrust::device_pointer_cast<int4>((int4 *)blob_stats_raw_ptr);

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
    thrust::transform_iterator blob_inds_iter(blob_inds_ptr, make_pos_from_index_op);

    Add_T_to_Blob_stat add_T_to_Blob_stat_op(T);
    thrust::transform_output_iterator  blob_stats_iter(blob_stats_ptr, add_T_to_Blob_stat_op);


    thrust::equal_to<int> binary_pred;

    auto reduce_out_iter = thrust::reduce_by_key(blob_labels_ptr, blob_labels_ptr + n_points, blob_inds_iter,
                                                 thrust::make_discard_iterator(), blob_stats_iter,
                                                 binary_pred, Add_Blob_stat());

    size_t n_blobs = thrust::get<1>(reduce_out_iter) - blob_stats_ptr;
    CC_labeled_image_ptr += img_len;
    blob_stats_ptr += n_blobs;
  }

  return std::make_tuple(blob_stats);
  ;
}*/

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
  // int *n_blobs_ptr = n_blobs.data_ptr<int>();
  //  thrust::device_ptr<int> blob_labels_ptr = thrust::device_pointer_cast<int>(blob_lables.data_ptr<int>());

  torch::Tensor blob_stats = torch::empty({input_sizes[0], (img_len / 4 + 1), 3}, output_tensor_options);
  // thrust::device_ptr<int3> blob_stats_ptr = thrust::device_pointer_cast<int3>(blob_stats.data_ptr<int3>());
  // thrust::device_ptr<Blob_stat> blob_stats_ptr = thrust::device_pointer_cast<Blob_stat>(blob_stats.data_ptr<Blob_stat>());
  int *blob_stats_raw_ptr = blob_stats.data_ptr<int>();
  thrust::device_ptr<Blob_stat> blob_stats_ptr = thrust::device_pointer_cast<Blob_stat>((Blob_stat *)blob_stats_raw_ptr);

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
    // thrust::transform_iterator<Make_pos_from_index, thrust::device_ptr<int> > blob_inds_iter(blob_inds_ptr, make_pos_from_index_op);
    thrust::transform_iterator blob_inds_iter(blob_inds_ptr, make_pos_from_index_op);

    auto reduce_out_iter = thrust::reduce_by_key(blob_labels_ptr, blob_labels_ptr + n_points, blob_inds_iter, thrust::make_discard_iterator(), blob_stats_ptr);

    n_blobs[T] = thrust::get<1>(reduce_out_iter) - blob_stats_ptr;
    CC_labeled_image_ptr += img_len;
    blob_stats_ptr += img_len / 4 + 1;
  }

  return std::make_tuple(blob_stats, n_blobs);
  ;
}


auto CC_Blob_Centers_Formatted(torch::Tensor &CC_labeled_image, float max_blob_ratio)
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



  thrust::device_ptr<int> CC_labeled_image_ptr = thrust::device_pointer_cast<int>(CC_labeled_image.data_ptr<int>());

  torch::Tensor blob_inds = torch::empty(img_len, output_tensor_options);
  thrust::device_ptr<int> blob_inds_ptr = thrust::device_pointer_cast<int>(blob_inds.data_ptr<int>());

  torch::Tensor blob_labels = torch::empty(img_len, output_tensor_options);
  thrust::device_ptr<int> blob_labels_ptr = thrust::device_pointer_cast<int>(blob_labels.data_ptr<int>());

  long max_blob_num = input_sizes[0]* (img_len / 4 + 1)*max_blob_ratio;
  torch::Tensor blob_stats = torch::empty({max_blob_num, 4}, output_tensor_options);

  int *blob_stats_raw_ptr = blob_stats.data_ptr<int>();
  thrust::device_ptr<Blob_stat4> blob_stats_ptr = thrust::device_pointer_cast<Blob_stat4>((Blob_stat4 *)blob_stats_raw_ptr);

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

    Make_pos_with_T_from_index make_pos_with_T_from_index_op(input_sizes[2],T);
    // thrust::transform_iterator<Make_pos_from_index, thrust::device_ptr<int> > blob_inds_iter(blob_inds_ptr, make_pos_from_index_op);
    thrust::transform_iterator blob_inds_iter(blob_inds_ptr, make_pos_with_T_from_index_op);

    auto reduce_out_iter = thrust::reduce_by_key(blob_labels_ptr, blob_labels_ptr + n_points, blob_inds_iter, thrust::make_discard_iterator(), blob_stats_ptr);

    size_t n_blobs = thrust::get<1>(reduce_out_iter) - blob_stats_ptr;
    CC_labeled_image_ptr += img_len;
    blob_stats_ptr += n_blobs;
  }

  return blob_stats;
  
}

PYBIND11_MODULE(CCL, m)
{
  m.def("CCL", &CCL, "CCL");
  m.def("CC_Blob_Centers", &CC_Blob_Centers, "CC_Blob_Centers");
  m.def("CC_Blob_Centers_Formatted", &CC_Blob_Centers_Formatted, "CC_Blob_Centers_Formatted");
  //m.def("CC_Blob_Centers_debug", &CC_Blob_Centers_debug, "CC_Blob_Centers_debug");
  // m.def("dummy", &dummy, "dummy desc");
  // m.def("sup_dude", &sup_dude, "suo dude desccc");
}

/*
auto CC_Blob_Centers_debug(torch::Tensor &CC_labeled_image)
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
  // int *n_blobs_ptr = n_blobs.data_ptr<int>();
  //  thrust::device_ptr<int> blob_labels_ptr = thrust::device_pointer_cast<int>(blob_lables.data_ptr<int>());

  torch::Tensor blob_stats = torch::empty({input_sizes[0], (img_len / 4 + 1), 3}, output_tensor_options);
  // thrust::device_ptr<int3> blob_stats_ptr = thrust::device_pointer_cast<int3>(blob_stats.data_ptr<int3>());
  // thrust::device_ptr<Blob_stat> blob_stats_ptr = thrust::device_pointer_cast<Blob_stat>(blob_stats.data_ptr<Blob_stat>());
  int *blob_stats_raw_ptr = blob_stats.data_ptr<int>();
  thrust::device_ptr<Blob_stat> blob_stats_ptr = thrust::device_pointer_cast<Blob_stat>((Blob_stat *)blob_stats_raw_ptr);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  double tot_copy_time = 0, tot_sort_time = 0, tot_reduce_time = 0;
  float cur_copy_time = 0, cur_sort_time = 0, cur_reduce_time = 0;

  for (int T = 0; T < input_sizes[0]; T++)
  {

    auto CC_input_zip_iter = thrust::make_zip_iterator(CC_labeled_image_ptr, thrust::make_counting_iterator(0));
    auto labels_zip_iter = thrust::make_zip_iterator(blob_labels_ptr, blob_inds_ptr);

    cudaEventRecord(start);
    auto n_points_iter = thrust::copy_if(CC_input_zip_iter,
                                         CC_input_zip_iter + img_len,
                                         labels_zip_iter,
                                         get_non_zero_labels());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cur_copy_time, start, stop);
    tot_copy_time += cur_copy_time;

    size_t n_points = n_points_iter - labels_zip_iter;

    cudaEventRecord(start);
    thrust::sort_by_key(blob_labels_ptr, blob_labels_ptr + n_points, blob_inds_ptr);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cur_sort_time, start, stop);
    tot_sort_time+=cur_sort_time;

    Make_pos_from_index make_pos_from_index_op(input_sizes[2]);
    // thrust::transform_iterator<Make_pos_from_index, thrust::device_ptr<int> > blob_inds_iter(blob_inds_ptr, make_pos_from_index_op);
    thrust::transform_iterator blob_inds_iter(blob_inds_ptr, make_pos_from_index_op);

    cudaEventRecord(start);
    auto reduce_out_iter = thrust::reduce_by_key(blob_labels_ptr, blob_labels_ptr + n_points, blob_inds_iter, thrust::make_discard_iterator(), blob_stats_ptr);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cur_reduce_time, start, stop);
    tot_reduce_time += cur_reduce_time;

    n_blobs[T] = thrust::get<1>(reduce_out_iter) - blob_stats_ptr;
    CC_labeled_image_ptr += img_len;
    blob_stats_ptr += img_len / 4 + 1;
  }

  std::cout << "tot_copy_time " << tot_copy_time << std::endl
            << "tot_sort_time " << tot_sort_time << std::endl 
            << "tot_reduce_time " <<  tot_reduce_time << std::endl;
  return std::make_tuple(blob_stats, n_blobs);
  ;
}
*/
