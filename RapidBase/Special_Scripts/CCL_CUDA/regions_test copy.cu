

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
#include "regions_bind.cu"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


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
  tst.index_put_({0, Slice(), Slice(None, None,3)}, 0);
  tst.index_put_({0,	Slice(None, None,3), Slice()}, 0);

  tst.index_put_({1, Slice(), Slice(None, None,3)}, 0);
  
  tst.index_put_({2,	Slice(None, None,3), Slice()}, 0);


  //std::cout << "TST DATA PTR outer" << tst.data_ptr<bool>() << std::endl;

  //std::cout << "tick" << std::endl;
  //start_point = high_resolution_clock::now(); // storing the starting time point in start 
  auto CC_labeled_image = CCL(tst);
  //end_point = high_resolution_clock::now(); //storing the ending time in end 
  //std::cout << "tock" << std::endl;

  //auto start = time_point_cast<microseconds>(start_point).time_since_epoch().count(); 
	// casting the time point to microseconds and measuring the time since time epoch
	
	//auto end = time_point_cast<microseconds>(end_point).time_since_epoch().count();

  //std::cout << (end-start) << std::endl;
  //std::cout << tst.index({0,	Slice(), Slice()}) ;
  std::cout << CC_labeled_image.index({0,	Slice(), Slice()}) << std::endl;
  gpuErrchk(cudaDeviceSynchronize());

  auto [blob_stats, n_blobs] = CC_Blob_Centers(CC_labeled_image);

  //std::cout << blob_stats.index({0,	Slice()}) << std::endl ;
  //std::cout << n_blobs << std::endl;
  //std::cout << n_blobs.sizes() << std::endl << n_blobs[0].sizes() <<std::endl;
  //std::cout << blob_stats.sizes() << std::endl << blob_stats[0].sizes() <<std::endl;
  //std::cout << blob_stats[0] << std::endl << n_blobs[0] << std::endl;
  for (int blob_num=0; blob_num < n_blobs[0].item<int>() ;blob_num++){
    std::cout <<  "blob x mean " <<  blob_stats[0][blob_num*3+0].item<float>()/float(blob_stats[0][blob_num*3+2].item<float>()) <<
     " " << "blob y mean " <<  blob_stats[0][blob_num*3+1].item<float>()/float(blob_stats[0][blob_num*3+2].item<float>()) << std::endl;
  }

    std::cout << CC_labeled_image.index({1,	Slice(), Slice()}) << std::endl;
  gpuErrchk(cudaDeviceSynchronize());

  //std::cout << blob_stats.index({0,	Slice()}) << std::endl ;
  //std::cout << n_blobs << std::endl;
  //std::cout << n_blobs.sizes() << std::endl << n_blobs[0].sizes() <<std::endl;
  //std::cout << blob_stats.sizes() << std::endl << blob_stats[0].sizes() <<std::endl;
  //std::cout << blob_stats[0] << std::endl << n_blobs[0] << std::endl;
  for (int blob_num=0; blob_num < n_blobs[1].item<int>() ;blob_num++){
    std::cout <<  "blob x mean " <<  blob_stats[1][blob_num*3+0].item<float>()/float(blob_stats[1][blob_num*3+2].item<float>()) <<
     " " << "blob y mean " <<  blob_stats[1][blob_num*3+1].item<float>()/float(blob_stats[1][blob_num*3+2].item<float>()) << std::endl;
  }

      std::cout << CC_labeled_image.index({2,	Slice(), Slice()}) << std::endl;
  gpuErrchk(cudaDeviceSynchronize());

  //std::cout << blob_stats.index({0,	Slice()}) << std::endl ;
  //std::cout << n_blobs << std::endl;
  //std::cout << n_blobs.sizes() << std::endl << n_blobs[0].sizes() <<std::endl;
  //std::cout << blob_stats.sizes() << std::endl << blob_stats[0].sizes() <<std::endl;
  //std::cout << blob_stats[0] << std::endl << n_blobs[0] << std::endl;
  for (int blob_num=0; blob_num < n_blobs[2].item<int>() ;blob_num++){
    std::cout <<  "blob x mean " <<  blob_stats[2][blob_num*3+0].item<float>()/float(blob_stats[2][blob_num*3+2].item<float>()) <<
     " " << "blob y mean " <<  blob_stats[2][blob_num*3+1].item<float>()/float(blob_stats[2][blob_num*3+2].item<float>()) << std::endl;
  }

  return 0;
};
