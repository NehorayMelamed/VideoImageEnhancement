

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
#include <chrono>


//#include "conv_data_types.cu"
#include "CCL.cu"


int main() {

  	using namespace std::chrono; 
    using namespace torch::indexing;	
	
	time_point<high_resolution_clock> start_point, end_point; // creating time points


  //torch::Tensor tensor = torch::ones({8000, 500,500});
  // auto in_sz = tensor.sizes();

  auto options =
      torch::TensorOptions()
          .dtype(torch::kBool)
          .layout(torch::kStrided)
          .device(torch::kCUDA)
          .requires_grad(false);


  int i,j;
  std::cout << "Please enter an integer value: ";
  std::cin >> i;
  std::cout << "Please enter an integer value: ";
  std::cin >> j;
  // torch::Tensor tst = torch::empty(tensor.sizes(),options);
  torch::Tensor tst = torch::ones({3, i,j}, options);
  //tst.index_put_({3,	Slice(), Slice()}, 0);
  //tst.index_put_({Slice(),	6, Slice()}, 0);
  //std::cout << tst.index({Slice(),	Slice(), 0}) ;
  tst.index_put_({Slice(), Slice(), Slice(None, None,3)}, 0);
  tst.index_put_({Slice(),	Slice(None, None,3), Slice()}, 0);


  std::cout << "TST DATA PTR outer" << tst.data_ptr<bool>() << std::endl;

  std::cout << "tick" << std::endl;
  start_point = high_resolution_clock::now(); // storing the starting time point in start 
  auto out = CCL(tst);
  end_point = high_resolution_clock::now(); //storing the ending time in end 
  std::cout << "tock" << std::endl;

  auto start = time_point_cast<microseconds>(start_point).time_since_epoch().count(); 
	// casting the time point to microseconds and measuring the time since time epoch
	
	auto end = time_point_cast<microseconds>(end_point).time_since_epoch().count();

  std::cout << (end-start) << std::endl;
  //std::cout << tst.index({0,	Slice(), Slice()}) ;
  std::cout << out.index({0,	Slice(), Slice()}) ;
  gpuErrchk(cudaDeviceSynchronize());
  return 0;
};
