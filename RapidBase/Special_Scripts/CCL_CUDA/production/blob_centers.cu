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
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>


struct Blob_stat
{

    int3 Stats_;

    __host__ __device__ int X() const { return Stats_.x; };
    __host__ __device__ int Y() const { return Stats_.y; };
    __host__ __device__ int Mass() const { return Stats_.z; };

    __host__ __device__ Blob_stat() : Stats_(make_int3(0, 0, 0)){};
    __host__ __device__ Blob_stat(const int3 in_stat) : Stats_(in_stat){};
    __host__ __device__ Blob_stat(const int x, const int y) : Stats_(make_int3(x, y, 1)){};
    __host__ __device__ Blob_stat(const int x, const int y, const int m) : Stats_(make_int3(x, y, m)){};
    __host__ __device__ Blob_stat operator+(const Blob_stat b) const 
    {
        return make_int3(X() + b.X(), Y()+ b.Y(), Mass() + b.Mass());
    }

   // __host__ __device__ operator int3() const {return Stats_;};
};

struct Blob_stat4
{

    int4 Stats_;

    __host__ __device__ int T() const { return Stats_.x; };
    __host__ __device__ int X() const { return Stats_.y; };
    __host__ __device__ int Y() const { return Stats_.z; };
    __host__ __device__ int Mass() const { return Stats_.w; };

    __host__ __device__ Blob_stat4() : Stats_(make_int4(0,0, 0, 0)){};
    __host__ __device__ Blob_stat4(const int4 in_stat) : Stats_(in_stat){};
    __host__ __device__ Blob_stat4(const int3 in_stat, int T) : Stats_(make_int4(T,in_stat.x,in_stat.y,in_stat.z ) ){};
    __host__ __device__ Blob_stat4(const int T, const int x, const int y) : Stats_(make_int4(T,x, y, 1)){};
    __host__ __device__ Blob_stat4(const int T, const int x, const int y, const int m) : Stats_(make_int4(T,x, y, m)){};
    __host__ __device__ Blob_stat4 operator+(const Blob_stat4 b) const 
    {
        return make_int4(T(),X() + b.X(), Y()+ b.Y(), Mass() + b.Mass());
    }

   // __host__ __device__ operator int3() const {return Stats_;};
};

//__host__ __device__ int3 operator +(const int3 a, const int3 b) {return make_int3(a.x+b.x,a.y+b.y,a.z+b.z);};

struct get_non_zero_labels{

  __host__ __device__ bool operator()(const thrust::tuple<int,int> in_tup) const {

    return (thrust::get<0>(in_tup)>0);
  }
};

struct Make_pos_from_index{

  const int stride_;
  __host__ __device__ Make_pos_from_index(const int stride_in):stride_(stride_in){};
  __host__ __device__ Blob_stat operator()(const int index) const{
    
    return (Blob_stat(index%stride_, index/stride_, 1));
  }
};

struct Make_pos_with_T_from_index{

  const int stride_;
  const int T;
  __host__ __device__ Make_pos_with_T_from_index(const int stride_in, const int T_in ):stride_(stride_in),T(T_in){};
  __host__ __device__ Blob_stat4 operator()(const int index) const{
    
    return (Blob_stat4(T,index%stride_, index/stride_, 1));
  }
};




// struct Make_pos_from_index{

//   const int stride_;
//   __host__ __device__ Make_pos_from_index(const int stride_in):stride_(stride_in){};
//   __host__ __device__ Blob_stat operator()(const int index) const{
    
//     return (make_int3(index%stride_, index/stride_, 1));
//   }
// };