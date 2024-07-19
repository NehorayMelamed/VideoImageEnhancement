#include <boost/core/demangle.hpp>
#include <cuda_fp16.h>
#include <iostream>
#include <stdexcept>
#include <torch/torch.h>

// #define ADD_TENSOR_TP(OP, TP, ...)                     \
//   switch (TP) {                                        \
//   case torch::kBool:                                   \
//     OP<bool>(__VA_ARGS__);                             \
//     break;                                             \
//   case torch::kUInt8:                                  \
//     OP<uint8_t>(__VA_ARGS__);                          \
//     break;                                             \
//   case torch::kInt8:                                   \
//     OP<int8_t>(__VA_ARGS__);                           \
//     break;                                             \
//   case torch::kInt16:                                  \
//     OP<int16_t>(__VA_ARGS__);                          \
//     break;                                             \
//   case torch::kInt32:                                  \
//     OP<int32_t>(__VA_ARGS__);                          \
//     break;                                             \
//   case torch::kInt64:                                  \
//     OP<int64_t>(__VA_ARGS__);                          \
//     break;                                             \
//   case torch::kFloat16:                                \
//     OP<half>(__VA_ARGS__);                             \
//     break;                                             \
//   case torch::kFloat32:                                \
//     OP<_Float32>(__VA_ARGS__);                         \
//     break;                                             \
//   case torch::kFloat64:                                \
//     OP<_Float64>(__VA_ARGS__);                         \
//     break;                                             \
//   default:                                             \
//     throw std::runtime_error("You son of a straying woman deserving of punishment! \n torch c++ does not support this data type"); \
//   }

template <typename TP>
void tst_func(TP a) {
  std::cout << typeid(a).name() << " " << a << std::endl;
}

int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  // auto in_sz = tensor.sizes();

  auto options =
      torch::TensorOptions()
          .dtype(torch::kUInt8)
          .layout(torch::kStrided)
          .device(torch::kCUDA)
          .requires_grad(false);

  // torch::Tensor tst = torch::empty(tensor.sizes(),options);
  torch::Tensor tst = torch::ones(tensor.sizes(), options);
  auto a = tst.sizes()[0];
  auto b = tst.sizes();
  std::cout << boost::core::demangle(typeid(tst.sizes()).name()) << std::endl;

  AT_DISPATCH_ALL_TYPES(tst.type(), "tst_func", [&] {tst_func<scalar_t> (1);});

  std::cout << *b.begin() << " " << *(b.end()-1) << std::endl;
  

};
