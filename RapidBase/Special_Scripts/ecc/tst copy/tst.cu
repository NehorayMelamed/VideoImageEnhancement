#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

using std::cout, std::endl;
typedef thrust::tuple<int, int, int> int3_tup;
typedef thrust::tuple<int, int> int2_tup;

struct func : public thrust::unary_function<int, int3_tup> {
  __device__ auto operator()(int x) // difsq
  {
    int a = -x;
    int b = x;
    int c = 2 * x;
    return thrust::make_tuple(a, b, c);
  }
};

struct func2 {

  __device__ auto operator()(int3_tup x, int3_tup y) // difsq
  {
    int a = thrust::get<0>(x) + thrust::get<0>(y);
    int b = thrust::get<1>(x) + thrust::get<1>(y);
    int c = thrust::get<2>(x) + thrust::get<2>(y);
    return thrust::make_tuple(a, b, c);
  }
};

struct func3 : public thrust::unary_function<int3_tup, int3_tup> {
  __device__ auto operator()(int3_tup x) // difsq
  {
    int a = -thrust::get<0>(x);
    int b = thrust::get<1>(x);
    int c = 2 * thrust::get<2>(x);
    return thrust::make_tuple(a, b, c);
  }
};

struct func4 : public thrust::unary_function<int3_tup, int2_tup> {
  __device__ auto operator()(int3_tup x) // difsq
  {
    int a = -thrust::get<0>(x);
    int b = thrust::get<1>(x) + thrust::get<2>(x);
    return thrust::make_tuple(a, b);
  }
};

int main() {
  thrust::device_vector<int> in_v(3);
  in_v[0] = 1;
  in_v[1] = 2;
  in_v[2] = 3;

  thrust::device_vector<int> out_v1(3);
  thrust::device_vector<int> out_v2(3);
  thrust::device_vector<int> out_v3(3);
  // typedef these iterators for shorthand
  typedef thrust::device_vector<int>::iterator IntIterator;
  // typedef a tuple of these iterators
  typedef thrust::tuple<IntIterator, IntIterator, IntIterator> IteratorTuple;
  // typedef the zip_iterator of this tuple
  typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
  // finally, create the zip_iterator

  ZipIterator iter(thrust::make_tuple(out_v1.begin(), out_v2.begin(), out_v3.begin()));

  thrust::transform(in_v.begin(), in_v.end(), iter, func()); // in-place transformation

  thrust::copy(out_v1.begin(), out_v1.end(), std::ostream_iterator<int>(std::cout, " "));
  cout << endl;
  thrust::copy(out_v2.begin(), out_v2.end(), std::ostream_iterator<int>(std::cout, " "));
  cout << endl;
  thrust::copy(out_v3.begin(), out_v3.end(), std::ostream_iterator<int>(std::cout, " "));
  cout << endl;

  auto c = thrust::transform_reduce(in_v.begin(), in_v.end(), func(), thrust::make_tuple(0, 0, 0), func2());
  cout << thrust::get<0>(c) << " " << thrust::get<1>(c) << " " << thrust::get<2>(c) << endl;

  thrust::device_vector<int> out2_v1(3);
  thrust::device_vector<int> out2_v2(3);
  thrust::device_vector<int> out2_v3(3);

  ZipIterator iter2(thrust::make_tuple(out2_v1.begin(), out2_v2.begin(), out2_v3.begin()));

  thrust::copy(thrust::make_transform_iterator(in_v.begin(), func()),
               thrust::make_transform_iterator(in_v.end(), func()), iter2);

  thrust::copy(out2_v1.begin(), out2_v1.end(), std::ostream_iterator<int>(std::cout, " "));
  cout << endl;
  thrust::copy(out2_v2.begin(), out2_v2.end(), std::ostream_iterator<int>(std::cout, " "));
  cout << endl;
  thrust::copy(out2_v3.begin(), out2_v3.end(), std::ostream_iterator<int>(std::cout, " "));
  cout << endl;

  thrust::device_vector<int> in2_0(2), in2_1(2), in2_2(2);
  in2_0[0] = 1;
  in2_1[0] = 2;
  in2_2[0] = 3;

  in2_0[1] = 1 * 2;
  in2_1[1] = 2 * 2;
  in2_2[1] = 3 * 2;

  ZipIterator iter3(thrust::make_tuple(in2_0.begin(), in2_1.begin(), in2_2.begin()));
  ZipIterator iter4(thrust::make_tuple(in2_0.end(), in2_1.end(), in2_2.end()));

  auto iter5 = thrust::make_transform_iterator(iter3, func3());
  auto iter6 = thrust::make_transform_iterator(iter4, func3());

  thrust::device_vector<int> out2_0(2), out2_1(2), out2_2(2);
  ZipIterator iter7(thrust::make_tuple(out2_0.begin(), out2_1.begin(), out2_2.begin()));

  thrust::copy(iter5, iter6, iter7);

  cout << endl;
  thrust::copy(out2_0.begin(), out2_0.end(), std::ostream_iterator<int>(std::cout, " "));
  cout << endl;
  thrust::copy(out2_1.begin(), out2_1.end(), std::ostream_iterator<int>(std::cout, " "));
  cout << endl;
  thrust::copy(out2_2.begin(), out2_2.end(), std::ostream_iterator<int>(std::cout, " "));
  cout << endl;

  auto iter8 = thrust::make_transform_iterator(iter3, func4());
  auto iter9 = thrust::make_transform_iterator(iter4, func4());

  typedef thrust::tuple<IntIterator, IntIterator> IteratorTuple2;
  // typedef the zip_iterator of this tuple
  typedef thrust::zip_iterator<IteratorTuple2> ZipIterator2;
  thrust::device_vector<int> out3_0(2), out3_1(2);
  ZipIterator2 iter10(thrust::make_tuple(out3_0.begin(), out3_1.begin()));

  thrust::copy(iter8, iter9, iter10);

  cout << endl;
  thrust::copy(out3_0.begin(), out3_0.end(), std::ostream_iterator<int>(std::cout, " "));
  cout << endl;
  thrust::copy(out3_1.begin(), out3_1.end(), std::ostream_iterator<int>(std::cout, " "));
  cout << endl;

  // *iter;   // returns (0, 0.0f, 'a')
  // iter[0]; // returns (0, 0.0f, 'a')
  // iter[1]; // returns (1, 1.0f, 'b')
  // iter[2]; // returns (2, 2.0f, 'c')
  // thrust::get<0>(iter[2]); // returns 2
  // thrust::get<1>(iter[0]); // returns 0.0f
  // thrust::get<2>(iter[1]); // returns 'b'
  // iter[3] is an out-of-bounds error
};