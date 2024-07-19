

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
#include <torch/extension.h>
#include <torch/library.h>
#include <torch/torch.h>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

using std::cout;
using std::endl;
// #include "conv_data_types.cu"

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

// void sup_dude() {
//   printf("sup dude priutnf");
//   cout << "sup dudeeeeeee" << endl;
// }

// void dummy(torch::Tensor & outliers_tensor) {
//   cout << "yo yo yo " << endl;
// }

auto CCL(torch::Tensor &outliers_tensor) {
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
  uint8_t *input_tensor_raw_ptr = outliers_tensor.data_ptr<uint8_t>();

  int *CCL_tensor_raw_ptr = CCL_labels_torch.data_ptr<int>();
/*
  thrust::device_ptr<uint8_t> thr_in(input_tensor_raw_ptr);
  thrust::device_ptr<int> thr_out(CCL_tensor_raw_ptr);
  thrust::copy(thr_in, thr_in + input_sizes[1] * input_sizes[2], std::ostream_iterator<uint8_t>(std::cout, " "));
  std::cout << std::endl
            << std::flush;
  thrust::copy(thr_out, thr_out + input_sizes[1] * input_sizes[2], std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl
            << std::flush;

  std::cout << std::endl
            << std::flush;
  std::cout << std::endl
            << std::flush;
  std::cout << std::endl
            << std::flush;
  */

#pragma unroll
  for (int z = 0; z < input_sizes[0]; z++) {

    auto CCL_labels = cv::cuda::GpuMat(input_sizes[1], input_sizes[2], CV_32S, CCL_tensor_raw_ptr);
    auto input_image = cv::cuda::GpuMat(input_sizes[1], input_sizes[2], CV_8U, input_tensor_raw_ptr);
    cv::cuda::connectedComponents(input_image, CCL_labels, 8, CV_32S);

    CCL_tensor_raw_ptr += input_sizes[1] * input_sizes[2];
    input_tensor_raw_ptr += input_sizes[1] * input_sizes[2];
  }
  return CCL_labels_torch;
}

TORCH_LIBRARY(CCL, m) {
  m.def("CCL", &CCL);
  // m.def("dummy", &dummy, "dummy desc");
  // m.def("sup_dude", &sup_dude, "suo dude desccc");
}
