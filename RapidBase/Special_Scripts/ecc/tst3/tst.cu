#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>

#include <cstdlib>
#include <iostream>
#include <ctime>

#include "tensors.cu"

using std::cout, std::endl;

/*
struct func : public thrust::unary_function<int,int> {

  int y;
  func(int Y):y(Y){};

  __host__ __device__ auto operator()(int x) // difsq
  {
    return (x+y);
  }
}; */
using tup2 = thrust::tuple< int, int>;


struct func2 : public thrust::unary_function<int, tup2 > 
{
  int y;
  func2(int Y):y(Y){};

  __host__ __device__ auto operator()(int x) 
  {
    tup2 res = thrust::make_tuple<int,int>(x,x+y);
    //tup2 res(x,x+y);
    return  res;
  }
};


struct func3 : public thrust::unary_function<int,int> {
  __device__ int operator()(int x) // difsq
  {
    return (-x);
  }
};

struct func4 : public thrust::unary_function<thrust::tuple<int,int>,int> {
  __device__ int operator()(thrust::tuple<int,int> input) // difsq
  {
    return input.get<0>()+ input.get<1>();
  }
};

using IntIterator = thrust::device_vector<int>::iterator ;

using neg_IntIterator = thrust::transform_output_iterator<func3, IntIterator> ;

int main() {
  
  int shft = std::rand()%10;
  cout << shft << endl;
  
  thrust::counting_iterator<int> iter(0);
  auto iter2 = thrust::make_transform_iterator(iter,func2(shft));

  thrust::device_vector<int> out_v1(10);
  thrust::device_vector<int> out_v2(10);

  neg_IntIterator iter3(out_v1.begin(), func3() );
  auto iter4 = make_zip_iterator(iter3,out_v2.begin()); 
   //iter2(iter, func(shft)());
  //transform_iterator<square_root, FloatIterator> iter(v.begin(), square_root());

  //thrust::copy(iter2 , iter2+10, out_v1.begin() );
  thrust::copy(iter2 , iter2+10, iter4 );
  thrust::copy(out_v1.begin(), out_v1.end(), std::ostream_iterator<int>(std::cout, " "));
  cout << endl;
    thrust::copy(out_v2.begin(), out_v2.end(), std::ostream_iterator<int>(std::cout, " "));
  cout << endl;
  cout << endl;

  auto iter5 = make_zip_iterator(out_v1.begin(),out_v2.begin());
  thrust::copy(iter , iter+10, thrust::make_transform_output_iterator(iter5,func2(shft)) );

  thrust::copy(out_v1.begin(), out_v1.end(), std::ostream_iterator<int>(std::cout, " "));
  cout << endl;
  thrust::copy(out_v2.begin(), out_v2.end(), std::ostream_iterator<int>(std::cout, " "));
  cout << endl;

  auto iter6 = make_zip_iterator(iter,iter);
  auto iter7 = thrust::make_transform_iterator(iter6,func4());
  std::cout << thrust::reduce(iter7+1,iter7+3,0) << std::endl;



};