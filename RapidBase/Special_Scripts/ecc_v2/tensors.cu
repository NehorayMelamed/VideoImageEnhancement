#pragma once

#include <cuda_runtime.h>

#include <array>
#include <boost/core/demangle.hpp>
#include <cuda/std/functional>
// #include <cuda/std/type_traits>
#include <functional>
#include <iostream>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include <tuple>

#include "./lang_extentions/SFINE_concepts.hpp"
// #include "lang_extentions/SFINE_concepts.hpp"
//  #include "3D_nl_flt_type_def.cpp"
//  #include <cuda/std/tuple>
//  #include <cuda/std/type_traits>
//  #include "3D_nl_globals.cu"

/*
#ifdef REALTYPE

typedef REALTYPE MAT_DEFAULT_TYPE ;

#else

typedef float MAT_DEFAULT_TYPE ;

#endif
* */

enum mat_sym_tp {
  none,
  sym
  //,anti_sym
}; // just an enum to help with matrix symmetry

template <class TP, int N, mat_sym_tp sym_tp>
class Mat_Base;

template <class TP, int N>
using Mat = Mat_Base<TP, N, mat_sym_tp::none>;

template <class TP>
using Mat2 = Mat<TP, 2>;

template <class TP>
using Mat3 = Mat<TP, 3>;

template <class TP, int N>
using Mat_Sym = Mat_Base<TP, N, mat_sym_tp::sym>;

template <class TP, int N>
using Mat_Sym = Mat_Base<TP, N, mat_sym_tp::sym>;

template <class TP>
using Mat2_Sym = Mat_Sym<TP, 2>;

template <class TP>
using Mat3_Sym = Mat_Sym<TP, 3>;

template <class TP, int N>
class Vec;

template <class TP>
using Vec2 = Vec<TP, 2>;

template <class TP>
using Vec3 = Vec<TP, 3>;

// template <class TP, int N>
// using Mat_Anti_Sym = Mat_Base<TP, N, mat_sym_tp::anti_sym>;

// template <class TP, int N>
// class Mat_Sym;
// template <class TP, int N>
// class Vec;

// template <class Mat>
// class Sub_vec;

namespace tensor_helper {

template <typename T>
struct ref_wrap {
  T &ref;
  typedef T type;

  __host__ __device__ inline explicit ref_wrap(T &val) : ref(val){};
  __host__ __device__ inline operator T &() {
    return ref;
  };
  __host__ __device__ inline T &operator=(const T &x) {
    ref = x;
    return ref;
  };
  __host__ __device__ inline operator T() const {
    return ref;
  };
};

template <typename T>
struct get_value_type_impl {
  using type = T;
};

template <typename T>
struct get_value_type_impl<std::reference_wrapper<T>> {
  using type = T;
};

template <typename T>
struct get_value_type_impl<ref_wrap<T>> {
  using type = T;
};

template <typename T>
using get_value_type = typename get_value_type_impl<T>::type;

template <std::size_t... I>
__host__ __device__ inline constexpr std::array<std::size_t, sizeof...(I)> create_ind_array_impl(std::index_sequence<I...>) {
  return std::array<std::size_t, sizeof...(I)>{{I...}};
}

template <std::size_t SZ>
__host__ __device__ inline constexpr std::array<std::size_t, SZ> create_ind_array() {
  return create_ind_array_impl(std::make_index_sequence<SZ>{});
}

template <size_t M, typename Z>
__host__ __device__ inline constexpr std::array<Z, M - 1> drop_ind(const std::array<Z, M> &arr_in, const int &ind) {

  std::array<std::size_t, M - 1> arr_out;
  int out_ind = 0;
  int in_ind = 0;

  for (; in_ind < ind; in_ind++, out_ind++)
    arr_out[out_ind] = arr_in[in_ind];

  for (in_ind++; in_ind < M; in_ind++, out_ind++)
    arr_out[out_ind] = arr_in[in_ind];

  return arr_out;
}

}; // namespace tensor_helper

template <class TP, int N, mat_sym_tp sym_tp>
class Mat_Base {
public:
  // static constexpr int DATA_SZ = (sym_tp == mat_sym_tp::none) ? N * N :
  // ((sym_tp == mat_sym_tp::sym) ? (N + 1) * N / 2 : (N - 1) * N / 2);
  static constexpr int DATA_SZ = (sym_tp == mat_sym_tp::none) ? N * N : (N + 1) * N / 2;

private:
  std::array<TP, DATA_SZ> data_;

public:
  typedef tensor_helper::get_value_type<TP> value_type;
  // typedef std::remove_refence_t<value_type> raw_type;
  typedef std::remove_cv_t<std::remove_pointer_t<value_type>> data_type;

  __host__ __device__ inline auto begin() {
    return data_.begin();
  };

  __host__ __device__ inline auto end() {
    return data_.end();
  };

  __host__ __device__ inline auto cbegin() const {
    return data_.cbegin();
  };

  __host__ __device__ inline auto cend() const {
    return data_.cend();
  };

  __host__ __device__ inline TP *Data() {
    return &data_[0];
  }
  __host__ __device__ inline const TP *Data() const {
    return &data_[0];
  }
  // typedef cuda::std::remove_cv_t<cuda::std::remove_pointer_t<value_type>>
  // data_type;

  __host__ __device__ inline TP &operator[](const int &n) {
    return data_[n];
  }

  __host__ __device__ inline TP operator[](const int &n) const {
    return data_[n];
  }

  /*  template <mat_sym_tp sym_tp_temp = sym_tp, std::enable_if_t<sym_tp_temp == mat_sym_tp::none, bool> = true>
    __host__ __device__ inline TP &operator()(const int &i, const int &j) {
      return data_[i * N + j];

        template <mat_sym_tp sym_tp_temp = sym_tp, std::enable_if_t<sym_tp_temp == mat_sym_tp::sym, bool> = true>
    __host__ __device__ inline TP &operator()(const int &i, const int &j) {
      auto calc_ind = [](const int &i, const int &j) { return ((i) * (i + 1) / 2 + j); };
      int ind = (j <= i) ? calc_ind(i, j) : calc_ind(j, i);
      return data_[ind];
    }
    }*/

  __host__ __device__ inline TP &operator()(const int &i, const int &j) {

    if constexpr (sym_tp == mat_sym_tp::none) {
      return data_[i * N + j];
    } else if constexpr (sym_tp == mat_sym_tp::sym) {
      auto calc_ind = [](const int &i, const int &j) {
        return ((i) * (i + 1) / 2 + j);
      };
      int ind = (j <= i) ? calc_ind(i, j) : calc_ind(j, i);
      return data_[ind];
    }
  }

  /*
    template <mat_sym_tp sym_tp_temp = sym_tp, std::enable_if_t<sym_tp_temp == mat_sym_tp::none, bool> = true>
    __host__ __device__ inline const TP operator()(const int &i,
                                                   const int &j) const {
      return data_[i * N + j];
    }

    template <mat_sym_tp sym_tp_temp = sym_tp, std::enable_if_t<sym_tp_temp == mat_sym_tp::sym, bool> = true>
    __host__ __device__ inline const TP operator()(const int &i, const int &j) const {
      auto calc_ind = [](const int &i, const int &j) { return ((i) * (i + 1) / 2 + j); };
      int ind = (j <= i) ? calc_ind(i, j) : calc_ind(j, i);
      return data_[ind];
    }*/

  __host__ __device__ inline TP operator()(const int &i, const int &j) const {

    if constexpr (sym_tp == mat_sym_tp::none) {
      return data_[i * N + j];
    } else if constexpr (sym_tp == mat_sym_tp::sym) {
      auto calc_ind = [](const int &i, const int &j) {
        return ((i) * (i + 1) / 2 + j);
      };
      int ind = (j <= i) ? calc_ind(i, j) : calc_ind(j, i);
      return data_[ind];
    }
  }

  /*   template <mat_sym_tp sym_tp_temp = sym_tp, typename
     std::enable_if_t<sym_tp_temp == mat_sym_tp::anti_sym, bool> = true>
     __host__ __device__ inline TP &operator()(const int &i, const int &j)
     {

         int calc_ind = [](const int &i, const int & j) { (i-1)*(i)/2 +  j  ;
     };


         else if (j<i)
         return data_[ calc_ind(i,j)];
         else if (j>i)
         return data_[ calc_ind(j,i)];
         else
         return 0; // this is a problem
     }*/

  __host__ __device__ inline Mat_Base(){}; // V

  __host__ __device__ inline explicit Mat_Base(TP val) { // V

#pragma unroll
    for (int a = 0; a < DATA_SZ; a++) {
      data_[a] = val;
    }
  };

  template <typename... Ts, std::enable_if_t<are_all_conv_to_T_v<TP, Ts...> and has_SZ<DATA_SZ, Ts...>, bool> = true>
  __host__ __device__ inline Mat_Base(Ts &&...ts) : data_{TP(ts)...} {}; // V

  __host__ __device__ inline Mat_Base(const std::array<TP, DATA_SZ> &arr) // V
      : data_(arr){};

  __host__ __device__ inline explicit Mat_Base(const Vec<TP, N> &diag) {

    Mat_Base &mat = *this;

#pragma unroll
    for (int i = 0; i < N; i++) {
      mat(i, i) = diag(i);
    }

    if constexpr (sym_tp == mat_sym_tp::sym) {
#pragma unroll
      for (int i = 0; i < N; i++) {
#pragma unroll
        for (int j = i + 1; j < N; j++) {
          mat(i, j) = TP(0);
        }
      }
    } else if constexpr (sym_tp == mat_sym_tp::none) {
#pragma unroll
      for (int i = 0; i < N; i++) {
#pragma unroll
        for (int j = i + 1; j < N; j++) {
          mat(i, j) = TP(0);
          mat(j, i) = TP(0);
        }
      }
    }
  };

  __host__ __device__ inline Mat_Base(const std::array<std::array<TP, N>, N> &mat_in) { // V
    Mat_Base &mat = *this;
#pragma unroll
    for (int i = 0; i < N; i++) {
#pragma unroll
      for (int j = 0; j < N; j++) {
        mat(i, j) = mat_in[i][j];
      }
    }
  };

  template <mat_sym_tp sym_tp_b = sym_tp, std::enable_if_t<sym_tp_b == mat_sym_tp::none, bool> = true>
  __host__ __device__ inline Mat_Base(const TP mat_in[N][N]) { // V
    Mat_Base &mat = *this;
#pragma unroll
    for (int i = 0; i < N; i++) {
#pragma unroll
      for (int j = 0; j < N; j++) {
        mat(i, j) = mat_in[i][j];
      }
    }
  };

  template <class U, mat_sym_tp sym_tp_a = sym_tp, std::enable_if_t<sym_tp_a == mat_sym_tp::none, bool> = true>
  __host__ __device__ inline Mat_Base(const Mat_Base<U, N, mat_sym_tp::sym> mat_in) { // V
    Mat_Base &mat = *this;
#pragma unroll
    for (int i = 0; i < N; i++) {
#pragma unroll
      for (int j = 0; j < N; j++) {
        mat(i, j) = mat_in(i, j);
      }
    }
  };

  template <class U, int M, mat_sym_tp sym_tp_b,
            std::enable_if_t<(M < N) and ((sym_tp_b == sym_tp) or (sym_tp == mat_sym_tp::none)), bool> = true> // V
  __host__ __device__ inline explicit Mat_Base(const Mat_Base<U, M, sym_tp_b> mat_in) {                        // V
    Mat_Base &mat = *this;

    if constexpr (sym_tp == mat_sym_tp::sym) {

#pragma unroll
      for (int i = 0; i < N; i++) {
#pragma unroll
        for (int j = i; j < N; j++) { // need to fix
          if ((j < M) and (i < M)) {
            mat(i, j) = mat_in(i, j);
          } else {
            mat(i, j) = TP(0);
          }
        }
      }
    } else if constexpr (sym_tp == mat_sym_tp::none) {
#pragma unroll
      for (int i = 0; i < N; i++) {
#pragma unroll
        for (int j = 0; j < N; j++) {
          if ((j < M) and (i < M)) { // need to fix
            mat(i, j) = mat_in(i, j);
          } else {
            mat(i, j) = TP(0);
          }
        }
      }
    }
  };

  __host__ __device__ inline Mat_Base(const std::initializer_list<TP> list) { // V

    auto it = list.begin();
#pragma unroll
    for (int ind = 0; ind < DATA_SZ; ind++, it++) {
      (*this)[ind] = *it;
    }
  };

  __host__ __device__ inline Mat_Base(const std::initializer_list<std::initializer_list<TP>> list) { // V
    Mat_Base &mat = *this;
    auto row_it = list.begin();

#pragma unroll
    for (int i = 0; i < N; i++, row_it++) {
      auto col_it = (*row_it).begin();
#pragma unroll
      for (int j = 0; j < N; j++, col_it++) {
        mat(i, j) = *col_it;
      }
    }
  };

  __host__ __device__ inline TP &Transpose(const int &i, const int &j) { // V
    return (*this)(j, i);
  }

  __host__ __device__ inline TP Transpose(const int &i, const int &j) const { // V
    return (*this)(j, i);
  }

  __host__ __device__ inline auto Transpose() const { // V
    const Mat_Base &mat = *this;
    Mat_Base<TP, N, sym_tp> trans_mat;

    if constexpr (sym_tp == mat_sym_tp::none) {
#pragma unroll
      for (int i = 0; i < N; i++) {
#pragma unroll
        for (int j = 0; j < N; j++) {
          trans_mat(i, j) = mat(j, i);
        }
      }

    } else if constexpr (sym_tp == mat_sym_tp::sym) { // V
      trans_mat = mat;
    }

    return trans_mat;
  }

  // std::enable_if_t<std::is_integral<TP>::value, bool> = true>
  __host__ __device__ inline bool Is_Sym() const { // V
    const Mat_Base &mat = *this;

    if constexpr (sym_tp == mat_sym_tp::none) {

#pragma unroll
      for (int i = 0; i < N; i++) {
#pragma unroll
        for (int j = i + 1; j < N; j++) {
          if (mat(i, j) != mat(j, i)) {
            return false;
          }
        }
      }
    }

    return true;
  }

  __host__ __device__ inline bool Is_Skew() const { // v
    const Mat_Base &mat = *this;
#pragma unroll
    for (int i = 0; i < N; i++) {
#pragma unroll
      for (int j = i + 1; j < N; j++) {
        if (mat(i, j) != (-mat(j, i))) {
          return false;
        }
      }
    }

#pragma unroll
    for (int i = 0; i < N; i++) {
      if (mat(i, i) != 0) {
        return false;
      }
    }

    return true;
  }

  __host__ __device__ inline static Mat_Base<value_type, N, sym_tp> Identity(int val = 1) { // v
    Mat_Base<value_type, N, sym_tp> mat;

    if constexpr (sym_tp == mat_sym_tp::none) {
#pragma unroll
      for (int i = 0; i < N; i++) {
#pragma unroll
        for (int j = i + 1; j < N; j++) {
          mat(i, j) = TP(0);
          mat(j, i) = TP(0);
        }
      }
    } else if constexpr (sym_tp == mat_sym_tp::sym) {
#pragma unroll
      for (int i = 0; i < N; i++) {
#pragma unroll
        for (int j = i + 1; j < N; j++) {
          mat(i, j) = TP(0);
        }
      }
    }

#pragma unroll
    for (int i = 0; i < N; i++) {
      mat(i, i) = val;
    }

    return mat;
  }

  __host__ __device__ inline static Mat_Base<value_type, N, mat_sym_tp::none> Axis_Rot(int ind1, int ind2, value_type angle) { // v

    // Mat_Base<value_type, N, mat_sym_tp::none> = mat(0);
    Mat_Base<value_type, N, mat_sym_tp::none> mat = Identity(1);
    mat(ind1, ind1) = std::cos(angle);
    mat(ind2, ind2) = mat(ind1, ind1);

    mat(ind2, ind1) = std::sin(angle);
    mat(ind1, ind2) = -mat(ind2, ind1);

    return mat;
  }

  __host__ __device__ inline static Mat_Base<value_type, N, mat_sym_tp::none> Axis_Rot_deg(int ind1, int ind2, value_type angle) { // v

    return Axis_Rot(ind1, ind2, angle * (atan(1) * 4) / 180);
  }

  // template <std::enable_if_t<N == 3, bool> = true>
  __host__ __device__ inline static Mat_Base<value_type, 3, mat_sym_tp::none> Axis_Rot_deg(int ind, value_type angle) { // v

    Mat_Base<value_type, 3, mat_sym_tp::none> mat;
    switch (ind) {
    case 0:
      mat = Axis_Rot_deg(1, 2, angle);
      break;
    case 1:
      mat = Axis_Rot_deg(2, 0, angle);
      break;
    case 2:
      mat = Axis_Rot_deg(0, 1, angle);
    }

    return mat;
  }

  //  any two ideintcal mats;
  /*
  template <typename U, mat_sym_tp sym_tp_rhs, std::enable_if_t<sym_tp_rhs == sym_tp, bool> = true>
  __host__ __device__ inline auto &operator=(const Mat_Base<U, N, sym_tp> &rhs) {
    auto &mat = *this;
    auto a = mat.begin();
    auto b = rhs.begin();
  #pragma unroll
    for (; a < a.end(); a++, b++) {
      (*a) = (*b);
    }

    return *this;
  }

  //  sym_mat to mat;
  template <typename U, mat_sym_tp sym_tp_rhs, std::enable_if_t<(sym_tp_rhs != sym_tp), bool> = true>
  __host__ __device__ inline auto &operator=(const Mat_Base<U, N, mat_sym_tp::sym> &rhs) {
    Mat_Base &mat = *this;

  #pragma unroll
    for (int i = 0; i < N; i++) {
  #pragma unroll
      for (int j = 0; j < N; j++) {
        mat(i, j) = rhs(i, j);
      }
    }
    return *this;
  }*/

  template <typename U, mat_sym_tp sym_tp_rhs, std::enable_if_t<((sym_tp_rhs == sym_tp) or (sym_tp == mat_sym_tp::none)), bool> = true>
  __host__ __device__ inline auto &operator=(const Mat_Base<U, N, sym_tp_rhs> &rhs) { // V

    Mat_Base &mat = *this;

    if constexpr (sym_tp_rhs == sym_tp) {
      // auto a = mat.begin();
      // auto b = rhs.begin();
#pragma unroll
      for (int ind = 0; ind < DATA_SZ; ind++) {
        mat[ind] = rhs[ind];
      }
    } else if constexpr (sym_tp_rhs != sym_tp) {

#pragma unroll
      for (int i = 0; i < N; i++) {
#pragma unroll
        for (int j = 0; j < N; j++) {
          mat(i, j) = rhs(i, j);
        }
      }
    }

    return *this;
  }

  /*
    template <typename U, mat_sym_tp sym_tp_rhs, std::enable_if_t<sym_tp_rhs == sym_tp, bool> = true>
    __host__ __device__ inline auto &operator+=(const Mat_Base<U, N, sym_tp> &rhs) {
      Mat_Base &mat = *this;
      auto a = mat.begin();
      auto b = rhs.begin();
  #pragma unroll
      for (; a < a.end(); a++, b++) {
        (*a) += (*b);
      }

      return *this;
    }

    //  sym_mat to mat;
    template <typename U, mat_sym_tp sym_tp_rhs, std::enable_if_t<(sym_tp_rhs != sym_tp), bool> = true>
    __host__ __device__ inline auto &operator+=(const Mat_Base<U, N, mat_sym_tp::sym> &rhs) {
      Mat_Base &mat = *this;

  #pragma unroll
      for (int i = 0; i < N; i++) {
  #pragma unroll
        for (int j = 0; j < N; j++) {
          mat(i, j) += rhs(i, j);
        }
      }
      return *this;
    }*/

  template <typename U, mat_sym_tp sym_tp_rhs, std::enable_if_t<((sym_tp_rhs == sym_tp) or (sym_tp == mat_sym_tp::none)), bool> = true>
  __host__ __device__ inline auto &operator+=(const Mat_Base<U, N, sym_tp_rhs> &rhs) { // V
    Mat_Base &mat = *this;
    if constexpr (sym_tp == sym_tp_rhs) {
      // auto a = mat.begin();
      // auto b = rhs.begin();
#pragma unroll
      for (int a = 0; a < DATA_SZ; a++) {
        mat[a] += rhs[a];
      }
    } else if constexpr (sym_tp_rhs != sym_tp) {
#pragma unroll
      for (int i = 0; i < N; i++) {
#pragma unroll
        for (int j = 0; j < N; j++) {
          mat(i, j) += rhs(i, j);
        }
      }
    }

    return *this;
  }

  /*
    template <typename U, mat_sym_tp sym_tp_rhs, std::enable_if_t<sym_tp_rhs == sym_tp, bool> = true>
    __host__ __device__ inline auto &operator-=(const Mat_Base<U, N, sym_tp> &rhs) {
      Mat_Base &mat = *this;
      auto a = mat.begin();
      auto b = rhs.begin();
  #pragma unroll
      for (; a < a.end(); a++, b++) {
        (*a) -= (*b);
      }

      return *this;
    }

    //  sym_mat to mat;
    template <typename U, mat_sym_tp sym_tp_rhs, std::enable_if_t<(sym_tp_rhs != sym_tp), bool> = true>
    __host__ __device__ inline auto &operator-=(const Mat_Base<U, N, mat_sym_tp::sym> &rhs) {
      Mat_Base &mat = *this;

  #pragma unroll
      for (int i = 0; i < N; i++) {
  #pragma unroll
        for (int j = 0; j < N; j++) {
          mat(i, j) -= rhs(i, j);
        }
      }
      return *this;
    }*/

  template <typename U, mat_sym_tp sym_tp_rhs, std::enable_if_t<((sym_tp_rhs == sym_tp) or (sym_tp == mat_sym_tp::none)), bool> = true>
  __host__ __device__ inline auto &operator-=(const Mat_Base<U, N, sym_tp_rhs> &rhs) { // V
    Mat_Base &mat = *this;
    if constexpr (sym_tp == sym_tp_rhs) {
      // auto a = mat.begin();
      // auto b = rhs.begin();
#pragma unroll
      for (int a = 0; a < DATA_SZ; a++) {
        mat[a] -= rhs[a];
      }
    } else if constexpr (sym_tp_rhs != sym_tp) {
#pragma unroll
      for (int i = 0; i < N; i++) {
#pragma unroll
        for (int j = 0; j < N; j++) {
          mat(i, j) -= rhs(i, j);
        }
      }
    }

    return *this;
  }
  /*
    template <typename U, mat_sym_tp sym_tp_rhs, std::enable_if_t<sym_tp_rhs == sym_tp, bool> = true>
    __host__ __device__ inline bool operator==(const Mat_Base<U, N, sym_tp> &rhs) const {
      Mat_Base &mat = *this;

      auto a = mat.begin();
      auto b = rhs.begin();
  #pragma unroll
      for (; a < a.end(); a++, b++) {
        if ((*a) != (*b))
          return false;
      }
      return true;
    }

    template <typename U, mat_sym_tp sym_tp_rhs, std::enable_if_t<(sym_tp_rhs != sym_tp), bool> = true>
    __host__ __device__ inline bool operator==(const Mat_Base<U, N, sym_tp_rhs> &rhs) {
      Mat_Base &mat = *this;

  #pragma unroll
      for (int i = 0; i < N; i++) {
  #pragma unroll
        for (int j = 0; j < N; j++) {
          if (mat(i, j) != rhs(i, j))
            return false;
        }
      }
      return true;
    }*/

  template <typename U, mat_sym_tp sym_tp_rhs>
  __host__ __device__ inline bool operator==(const Mat_Base<U, N, sym_tp_rhs> &rhs) const { // V

    const Mat_Base &mat = *this;

    if constexpr (sym_tp == sym_tp_rhs) {
      // auto a = mat.begin();
      // auto b = rhs.begin();
#pragma unroll
      for (int ind = 0; ind < DATA_SZ; ind++) {
        if (mat[ind] != mat[ind])
          return false;
      }
    } else if constexpr (sym_tp_rhs != sym_tp) {

#pragma unroll
      for (int i = 0; i < N; i++) {
#pragma unroll
        for (int j = 0; j < N; j++) {
          if (mat(i, j) != rhs(i, j))
            return false;
        }
      }
    }

    return true;
  }

  template <typename U, std::enable_if_t<are_addble_as_T_v<value_type, U>, bool> = true>
  __host__ __device__ inline auto &operator+=(const U &d) { // V
    Mat_Base &mat = *this;
    // auto a = mat.begin();
#pragma unroll
    for (int ind = 0; ind < DATA_SZ; ind++) {
      mat[ind] += d;
    }

    return *this;
  }

  template <typename U, std::enable_if_t<are_addble_as_T_v<value_type, U>, bool> = true>
  __host__ __device__ inline auto &operator-=(const U &d) { // V
    Mat_Base &mat = *this;
    // auto a = mat.begin();
#pragma unroll
    for (int ind = 0; ind < DATA_SZ; ind++) {
      mat[ind] -= d;
    }

    return *this;
  }

  template <typename U, std::enable_if_t<are_addble_as_T_v<value_type, U>, bool> = true>
  __host__ __device__ inline auto &operator*=(const U &d) { // V
    Mat_Base &mat = *this;
    // auto a = mat.begin();
#pragma unroll
    for (int ind = 0; ind < DATA_SZ; ind++) {
      mat[ind] *= d;
    }

    return *this;
  }

  template <typename U, std::enable_if_t<are_addble_as_T_v<value_type, U>, bool> = true>
  __host__ __device__ inline auto &operator/=(const U &d) { // V
    Mat_Base &mat = *this;
    // auto a = mat.begin();
#pragma unroll
    for (int ind = 0; ind < DATA_SZ; ind++) {
      mat[ind] /= d;
    }

    return *this;
  }

  template <typename U, mat_sym_tp sym_tp_rhs, mat_sym_tp sym_tp_lhs = sym_tp, std::enable_if_t<(sym_tp_lhs == mat_sym_tp::none), bool> = true>
  __host__ __device__ inline Mat_Base<TP, N, mat_sym_tp::none> &operator*=(const Mat_Base<U, N, sym_tp_rhs> &rhs) { // V
    Mat_Base &mat = *this;
    Mat_Base<TP, N, mat_sym_tp::none> mat_out;
#pragma unroll
    for (int i = 0; i < N; i++) {
#pragma unroll
      for (int j = 0; j < N; j++) {
        mat_out(i, j) = mat(i, 0) * rhs(0, j);
#pragma unroll
        for (int k = 1; k < N; k++) {
          mat_out(i, j) += mat(i, k) * rhs(k, j);
        }
      }
    }

    mat = mat_out;
    return mat;
  }
  /*
    template <typename U, mat_sym_tp sym_tp_b, std::enable_if_t<sym_tp_b == sym_tp, bool> = true>
    __host__ __device__ inline auto operator+(const Mat_Base<U, N, sym_tp_b> &b) const {
      const Mat_Base &a = *this;
      Mat_Base<decltype(a(0, 0) + b(0, 0)), N, sym_tp> c;

      auto a_ptr = a.begin();
      auto b_ptr = b.begin();
      auto c_ptr = c.begin();

  #pragma unroll
      for (; a_ptr < a.end(); a_ptr++, b_ptr++, c_ptr++) {
        (*c_ptr) = (*a_ptr) + (*b_ptr);
      }

      return c;
    }

    template <typename U, mat_sym_tp sym_tp_b, std::enable_if_t<sym_tp_b != sym_tp, bool> = true>
    __host__ __device__ inline auto operator+(const Mat_Base<U, N, sym_tp_b> &b) const {
      const Mat_Base &a = *this;
      Mat_Base<decltype(a(0, 0) + b(0, 0)), N, mat_sym_tp::none> c;

  #pragma unroll
      for (int i = 0; i < N; i++) {
  #pragma unroll
        for (int j = 0; j < N; i++) {
          c(i, j) = a(i, j) + b(i, j);
        }
      }

      return c;
    }*/

  template <typename U, mat_sym_tp sym_tp_b>
  __host__ __device__ inline auto operator+(const Mat_Base<U, N, sym_tp_b> &b) const { // V

    const Mat_Base &a = *this;
    Mat_Base<decltype(a(0, 0) + b(0, 0)), N, sym_tp_b == sym_tp ? sym_tp : mat_sym_tp::none> c;

    if constexpr (sym_tp == sym_tp_b) {
      // auto a_ptr = a.cbegin();
      // auto b_ptr = b.cbegin();
      // auto c_ptr = c.begin();

#pragma unroll
      for (int ind = 0; ind < DATA_SZ; ind++) {
        c[ind] = a[ind] + b[ind];
      }
    } else if constexpr (sym_tp_b != sym_tp) {

#pragma unroll
      for (int i = 0; i < N; i++) {
#pragma unroll
        for (int j = 0; j < N; j++) {
          c(i, j) = a(i, j) + b(i, j);
        }
      }
    }

    return c;
  }

  /*
    template <typename U, mat_sym_tp sym_tp_b, std::enable_if_t<sym_tp_b == sym_tp, bool> = true>
    __host__ __device__ inline auto operator-(const Mat_Base<U, N, sym_tp_b> &b) const {
      const Mat_Base &a = *this;
      Mat_Base<decltype(a(0, 0) + b(0, 0)), N, sym_tp> c;

      auto a_ptr = a.begin();
      auto b_ptr = b.begin();
      auto c_ptr = c.begin();

  #pragma unroll
      for (; a_ptr < a.end(); a_ptr++, b_ptr++, c_ptr++) {
        (*c_ptr) = (*a_ptr) - (*b_ptr);
      }

      return c;
    }

    template <typename U, mat_sym_tp sym_tp_b, std::enable_if_t<sym_tp_b != sym_tp, bool> = true>
    __host__ __device__ inline auto operator-(const Mat_Base<U, N, sym_tp_b> &b) const {
      const Mat_Base &a = *this;
      Mat_Base<decltype(a(0, 0) - b(0, 0)), N, mat_sym_tp::none> c;

  #pragma unroll
      for (int i = 0; i < N; i++) {
  #pragma unroll
        for (int j = 0; j < N; i++) {
          c(i, j) = a(i, j) - b(i, j);
        }
      }

      return c;
    }*/

  __host__ __device__ inline auto operator-() const { // V

    const Mat_Base &a = *this;
    Mat_Base< value_type , N, sym_tp> c;

#pragma unroll
      for (int ind = 0; ind < DATA_SZ; ind++) {
        c[ind] = -a[ind] ;
      }
    return c;
  }

  template <typename U, mat_sym_tp sym_tp_b>
  __host__ __device__ inline auto operator-(const Mat_Base<U, N, sym_tp_b> &b) const { // V

    const Mat_Base &a = *this;
    Mat_Base<decltype(a(0, 0) + b(0, 0)), N, sym_tp_b == sym_tp ? sym_tp : mat_sym_tp::none> c;

    if constexpr (sym_tp == sym_tp_b) {

      //      auto a_ptr = a.cbegin();
      //      auto b_ptr = b.cbegin();
      //      auto c_ptr = c.begin();

#pragma unroll
      for (int ind = 0; ind < DATA_SZ; ind++) {
        c[ind] = a[ind] - b[ind];
      }
    } else if constexpr (sym_tp_b != sym_tp) {

#pragma unroll
      for (int i = 0; i < N; i++) {
#pragma unroll
        for (int j = 0; j < N; j++) {
          c(i, j) = a(i, j) - b(i, j);
        }
      }
    }

    return c;
  }

  // template <typename Z, std::enable_if_t<(std::is_convertible_v<TP, Z>) or (std::is_convertible_v<Z, TP>), bool> = true>
  template <typename Z, std::enable_if_t<are_addble_v<TP, Z>, bool> = true>
  __host__ __device__ inline auto operator*(const Z &b) const { // v
    const Mat_Base &a = *this;
    Mat_Base<decltype(a(0, 0) + b), N, sym_tp> c;

    // auto a_ptr = a.cbegin();
    // auto c_ptr = c.begin();
#pragma unroll
    for (int ind = 0; ind < DATA_SZ; ind++) {
      c[ind] = a[ind] * b;
    }

    return c;
  }

  // template <typename Z, std::enable_if_t<(std::is_convertible_v<TP, Z>) or (std::is_convertible_v<Z, TP>), bool> = true>
  template <typename Z, std::enable_if_t<are_addble_v<TP, Z>, bool> = true>
  __host__ __device__ inline auto operator/(const Z &b) const { // v
    const Mat_Base &a = *this;
    Mat_Base<decltype(a(0, 0) + b), N, sym_tp> c;

    // auto a_ptr = a.cbegin();
    // auto c_ptr = c.begin();
#pragma unroll
    for (int ind = 0; ind < DATA_SZ; ind++) {
      c[ind] = a[ind] / b;
    }

    return c;
  }

  template <typename Z, mat_sym_tp sym_tp_b>
  __host__ __device__ inline auto operator*(const Mat_Base<Z, N, sym_tp_b> &b) const { // v
    const Mat_Base &a = *this;
    Mat_Base<decltype(a(0, 0) * b(0, 0)), N, mat_sym_tp::none> c;

    // printf(" %f %f \n %f %f \n\n", a(0,0), a(0,1) , a(1,0) , a(1,1) );
    // printf(" %f %f \n %f %f \n\n", b(0,0), b(0,1) , b(1,0) , b(1,1) );
#pragma unroll
    for (int i = 0; i < N; i++) {
#pragma unroll
      for (int j = 0; j < N; j++) {
        c(i, j) = a(i, 0) * b(0, j);
#pragma unroll
        for (int k = 1; k < N; k++) {
          c(i, j) += a(i, k) * b(k, j);
        }
      }
    }
    return c;
  }

  template <typename Z>
  __host__ __device__ inline auto operator*(const Vec<Z, N> &vec) const; // V

  __host__ __device__ inline Mat_Base<TP, N, mat_sym_tp::sym> Symmetric_Part() const { // v

    const Mat_Base &mat = *this;
    Mat_Base<TP, N, mat_sym_tp::sym> sym_part;

    if constexpr (sym_tp == mat_sym_tp::sym) {
      sym_part = mat;
    } else if constexpr (sym_tp == mat_sym_tp::none) {

#pragma unroll
      for (int i = 0; i < N; i++) {
#pragma unroll
        for (int j = i + 1; j < N; j++) {
          sym_part(i, j) = (mat(i, j) + mat(j, i)) / TP(2);
        }
      }

#pragma unroll
      for (int i = 0; i < N; i++) {
        sym_part(i, i) = mat(i, i);
      }
    }

    return sym_part;
  };

  __host__ __device__ inline Mat_Base<value_type, N, mat_sym_tp::sym> Metric() const { // V

    const Mat_Base mat = *this;
    Mat_Base<value_type, N, sym> metric;

#pragma unroll
    for (int i = 0; i < N; i++) {
#pragma unroll
      for (int j = i; j < N; j++) {
        metric(i, j) = mat(0, i) * mat(0, j);
#pragma unroll
        for (int k = 1; k < N; k++) {
          metric(i, j) += mat(k, i) * mat(k, j);
        }
      }
    }

    return metric;
  };

  __host__ __device__ inline value_type Trace() const { // V
    const Mat_Base &mat = *this;
    value_type sum = mat(0, 0);
#pragma unroll
    for (int ind = 1; ind < N; ind++) {
      sum += mat(ind, ind);
    }

    return sum;
  }

  template <typename Z, mat_sym_tp sym_tp_b>
  __host__ __device__ inline auto inner_prod(const Mat_Base<Z, N, sym_tp_b> &b) const { // V
    const Mat_Base &a = *this;
    decltype(a(0, 0) * b(0, 0)) prod = value_type(0.0);

    if constexpr ((sym_tp == mat_sym_tp::none) or (sym_tp_b == mat_sym_tp::none)) {
#pragma unroll
      for (int i = 0; i < N; i++) {
#pragma unroll
        for (int j = 0; j < N; j++) {
          prod += a(i, j) * b(i, j);
        }
      }
    } else if constexpr ((sym_tp == mat_sym_tp::sym) and (sym_tp_b == mat_sym_tp::sym)) {

#pragma unroll
      for (int i = 0; i < N; i++) {
#pragma unroll
        for (int j = i + 1; j < N; j++) {
          prod += value_type(2) * a(i, j) * b(i, j);
        }
      }

#pragma unroll
      for (int i = 0; i < N; i++) {
        prod += a(i, i) * b(i, i);
      }
    }

    return prod;
  }

  __host__ __device__ inline value_type Norm_Square() const { // v
    const Mat_Base &mat = *this;
    value_type mat_square = value_type(0.0);

    if constexpr (sym_tp == mat_sym_tp::none) {
#pragma unroll
      for (int i = 0; i < N; i++) {
#pragma unroll
        for (int j = 0; j < N; j++) {
          mat_square += mat(i, j) * mat(i, j);
        }
      }
    } else if constexpr (sym_tp == mat_sym_tp::sym) {

#pragma unroll
      for (int i = 0; i < N; i++) {
#pragma unroll
        for (int j = i + 1; j < N; j++) {
          mat_square += TP(2) * mat(i, j) * mat(i, j);
        }
      }

#pragma unroll
      for (int i = 0; i < N; i++) {
        mat_square += mat(i, i) * mat(i, i);
      }
    }
    return (mat_square);
  }

private:
  template <size_t M>
  __host__ __device__ value_type det_impel(const std::array<std::size_t, M> &rows, const std::array<std::size_t, M> &cols) const {

    namespace th = tensor_helper;

    const auto mat = *this;
    if constexpr (M == 1)
      return mat(rows[0], cols[0]);

    else {
      value_type sum = value_type(0);
      value_type minor, cur_elem;

#pragma unroll
      for (int iter_ind = 0; iter_ind < M; iter_ind++) {
        cur_elem = mat(rows[iter_ind], cols[0]);
        minor = det_impel<M - 1>(th::drop_ind(rows, iter_ind), th::drop_ind(cols, 0));
        ((iter_ind % 2) == 0) ? sum += minor *cur_elem : sum -= minor * cur_elem;
      }
      return sum;
    }
    return 0;
  }

public:
  __host__ __device__ inline value_type Det() const { // v

    namespace th = tensor_helper;
    // auto inds = create_ind_array<N>();
    constexpr std::array<std::size_t, N> inds = th::create_ind_array<N>();
    // auto  create_ind_array<std::size_t(N)>();
    value_type det = det_impel<N>(inds, inds);
    // TP det=1;
    return det;
  }

  __host__ __device__ inline Mat_Base<value_type, N, sym_tp> Adjoint() const { // V

    namespace th = tensor_helper;

    const auto &mat = *this;
    Mat_Base<value_type, N, sym_tp> adj_mat;
    constexpr std::array<std::size_t, N> inds = th::create_ind_array<N>();

    if constexpr (sym_tp == mat_sym_tp::none) {

#pragma unroll
      for (int i = 0; i < N; i++) {
#pragma unroll
        for (int j = 0; j < N; j++) {
          adj_mat(j, i) = std::pow(-1, i + j) * det_impel(th::drop_ind(inds, i), th::drop_ind(inds, j));
        }
      }

    } else if constexpr (sym_tp == mat_sym_tp::sym) {

#pragma unroll
      for (int i = 0; i < N; i++) {
#pragma unroll
        for (int j = i; j < N; j++) {
          adj_mat(j, i) = std::pow(-1, i + j) * det_impel(th::drop_ind(inds, i), th::drop_ind(inds, j));
        }
      }
    }

    return adj_mat;
  }

  __host__ __device__ inline std::pair<Mat_Base<value_type, N, sym_tp>, value_type> Adjoint_and_Det() const { // V

    const Mat_Base &mat = *this;
    Mat_Base<value_type, N, sym_tp> adj_mat = (*this).Adjoint();

    value_type det = adj_mat(0, 0) * mat(0, 0);
#pragma unroll
    for (int ind = 1; ind < N; ind++) {
      det += adj_mat(0, ind) * mat(ind, 0);
    }
    //+ adj_mat(0, 1) * mat(1, 0) + adj_mat(0, 2) * mat(2, 0);
    return {adj_mat, det};
  }

  /*
  template <mat_sym_tp sym_tp_b = sym_tp, std::enable_if_t<sym_tp_b == mat_sym_tp::none, bool> = true>
  __host__ __device__ inline Mat_Base<TP, N, sym_tp> Adjoint() const {
   const auto &mat = *this;
   Mat_Base<TP, N, sym_tp> adj_mat;
   constexpr std::array<int, N> inds(std::make_integer_sequence<std::size_t, N>);
  #pragma unroll
   for (int i = 0; i < N; i++) {
  #pragma unroll
     for (int j = 0; j < N; j++) {
       adj_mat(i, j) = std::pow(-1, i + j) * det_impel(drop_ind(inds, i), drop_ind(inds, j));
     }
   }
   return adj_mat;
  }

  template <mat_sym_tp sym_tp_b = sym_tp, std::enable_if_t<sym_tp_b != mat_sym_tp::sym, bool> = true>
  __host__ __device__ inline Mat_Base<TP, N, sym_tp> Adjoint() const {
   const auto &mat = *this;
   Mat_Base<TP, N, sym_tp> adj_mat;
   constexpr std::array<int, N> inds(std::make_integer_sequence<std::size_t, N>);
  #pragma unroll
   for (int i = 0; i < N; i++) {
  #pragma unroll
     for (int j = i; j < N; j++) {
       adj_mat(i, j) = std::pow(-1, i + j) * det_impel(drop_ind(inds, i), drop_ind(inds, j));
     }
   }
   return adj_mat;
  }*/

  /*

  template <std::size_t... inds>
  __host__ __device__ inline Vec<tensor_helper::ref_wrap<TP>, N>
  Mat_Base<TP, N, sym_tp>::Diag_impl(std::index_sequence<inds...>) { // V
   Mat_Base &mat = *this;
   return Vec<tensor_helper::ref_wrap<TP>, N>(tensor_helper::ref_wrap(mat(inds, inds))...);
  }


  __host__ __device__ inline Mat_Base<tensor_helper::ref ,N,sym_tp> Mat_Base<TP, N, sym_tp>::operator*() { // V
   // Mat_Base &mat = *this;
   return d_ref_impl(std::make_index_sequence<N>{});
  }
  */

  template <std::size_t... inds>
  __host__ __device__ inline Mat_Base<tensor_helper::ref_wrap<std::remove_pointer_t<value_type>>, N, sym_tp>
  d_ref_impl(std::index_sequence<inds...>) {
    Mat_Base &mat = *this;
    return Mat_Base<tensor_helper::ref_wrap<std::remove_pointer_t<value_type>>, N, sym_tp>(tensor_helper::ref_wrap(*(mat.data_[inds]))...);
  }; // v

  __host__ __device__ inline Mat_Base<tensor_helper::ref_wrap<std::remove_pointer_t<value_type>>, N, sym_tp>
  operator*() {
    return d_ref_impl(std::make_index_sequence<DATA_SZ>{});
  }; // v

  template <std::size_t... inds>
  __host__ __device__ inline Mat_Base<tensor_helper::ref_wrap<std::remove_pointer_t<value_type>>, N, sym_tp>
  d_ref_impl(std::index_sequence<inds...>) const {
    Mat_Base &mat = *this;
    return Mat_Base<tensor_helper::ref_wrap<std::remove_pointer_t<value_type>>, N, sym_tp>(tensor_helper::ref_wrap(*(mat.data_[inds]))...);
  }; // v

  __host__ __device__ inline Mat_Base<tensor_helper::ref_wrap<std::remove_pointer_t<value_type>>, N, sym_tp>
  operator*() const {
    return d_ref_impl(std::make_index_sequence<DATA_SZ>{});
  }; // v

  template <std::size_t... inds>
  __host__ __device__ inline Vec<tensor_helper::ref_wrap<TP>, N> Diag_impl(std::index_sequence<inds...>); // v

  __host__ __device__ inline Vec<tensor_helper::ref_wrap<TP>, N> Diag(); // v

  template <std::size_t... inds>
  __host__ __device__ inline Vec<tensor_helper::ref_wrap<TP>, N> Diag_impl(std::index_sequence<inds...>) const; // v

  __host__ __device__ inline Vec<tensor_helper::ref_wrap<TP>, N> Diag() const; // v

  template <std::size_t... inds>
  __host__ __device__ inline Vec<tensor_helper::ref_wrap<TP>, N> Row_impl(int row_num, std::index_sequence<inds...>); // v

  __host__ __device__ inline Vec<tensor_helper::ref_wrap<TP>, N> Row(int row_num);

  template <std::size_t... inds>
  __host__ __device__ inline Vec<tensor_helper::ref_wrap<TP>, N> Row_impl(int row_num, std::index_sequence<inds...>) const; // v

  __host__ __device__ inline Vec<tensor_helper::ref_wrap<TP>, N> Row(int row_num) const;

  template <std::size_t... inds>
  __host__ __device__ inline Vec<tensor_helper::ref_wrap<TP>, N> Col_impl(int col_num, std::index_sequence<inds...>); // v

  __host__ __device__ inline Vec<tensor_helper::ref_wrap<TP>, N> Col(int col_num);

  template <std::size_t... inds>
  __host__ __device__ inline Vec<tensor_helper::ref_wrap<TP>, N> Col_impl(int col_num, std::index_sequence<inds...>) const; // v

  __host__ __device__ inline Vec<tensor_helper::ref_wrap<TP>, N> Col(int col_num) const;

  class Elementwise_ {
  public:
    Mat_Base<TP, N, sym_tp> &mat_;

    __host__ __device__ Elementwise_(Mat_Base<TP, N, sym_tp> &mat_in) : mat_{mat_in} {};

    template <typename U, mat_sym_tp sym_tp_rhs, std::enable_if_t<((sym_tp_rhs == sym_tp) or (sym_tp == mat_sym_tp::none)), bool> = true>
    __host__ __device__ inline auto &operator*=(const Mat_Base<U, N, sym_tp_rhs> &rhs) { // V
      Mat_Base &mat = mat_;
      if constexpr (sym_tp == sym_tp_rhs) {
        // auto a = mat.begin();
        // auto b = rhs.begin();
#pragma unroll
        for (int ind = 0; ind < DATA_SZ; ind++) {
          mat[ind] *= rhs[ind];
        }
      } else if constexpr (sym_tp_rhs != sym_tp) {
#pragma unroll
        for (int i = 0; i < N; i++) {
#pragma unroll
          for (int j = 0; j < N; j++) {
            mat(i, j) *= rhs(i, j);
          }
        }
      }

      return mat;
    }

    template <typename U, mat_sym_tp sym_tp_rhs, std::enable_if_t<((sym_tp_rhs == sym_tp) or (sym_tp == mat_sym_tp::none)), bool> = true>
    __host__ __device__ inline auto &operator/=(const Mat_Base<U, N, sym_tp_rhs> &rhs) { // V
      Mat_Base &mat = mat_;
      if constexpr (sym_tp == sym_tp_rhs) {
        // auto a = mat.begin();
        // auto b = rhs.begin();
#pragma unroll
        for (int ind = 0; ind < DATA_SZ; ind++) {
          mat[ind] /= rhs[ind];
        }
      } else if constexpr (sym_tp_rhs != sym_tp) {
#pragma unroll
        for (int i = 0; i < N; i++) {
#pragma unroll
          for (int j = 0; j < N; j++) {
            mat(i, j) /= rhs(i, j);
          }
        }
      }

      return mat;
    }

    template <typename U, mat_sym_tp sym_tp_b>
    __host__ __device__ inline auto operator*(const Mat_Base<U, N, sym_tp_b> &b) const { // v

      const Mat_Base &a = mat_;
      Mat_Base<decltype(a(0, 0) + b(0, 0)), N, sym_tp_b == sym_tp ? sym_tp : mat_sym_tp::none> c;

      if constexpr (sym_tp == sym_tp_b) {

        // auto a_ptr = a.cbegin();
        // auto b_ptr = b.cbegin();
        // auto c_ptr = c.begin();

#pragma unroll
        for (int ind = 0; ind < DATA_SZ; ind++) {
          c[ind] = a[ind] * b[ind];
        }
      } else if constexpr (sym_tp_b != sym_tp) {

#pragma unroll
        for (int i = 0; i < N; i++) {
#pragma unroll
          for (int j = 0; j < N; j++) {
            c(i, j) = a(i, j) * b(i, j);
          }
        }
      }

      return c;
    }

    template <typename U, mat_sym_tp sym_tp_b>
    __host__ __device__ inline auto operator/(const Mat_Base<U, N, sym_tp_b> &b) const { // v

      const Mat_Base &a = mat_;
      Mat_Base<decltype(a(0, 0) + b(0, 0)), N, sym_tp_b == sym_tp ? sym_tp : mat_sym_tp::none> c;

      if constexpr (sym_tp == sym_tp_b) {

        /// auto a_ptr = a.cbegin();
        // auto b_ptr = b.cbegin();
        // auto c_ptr = c.begin();

#pragma unroll
        for (int ind = 0; ind < DATA_SZ; ind++) {
          c[ind] = a[ind] / b[ind];
        }
      } else if constexpr (sym_tp_b != sym_tp) {

#pragma unroll
        for (int i = 0; i < N; i++) {
#pragma unroll
          for (int j = 0; j < N; j++) {
            c(i, j) = a(i, j) / b(i, j);
          }
        }
      }

      return c;
    }
  };

  __host__ __device__ Elementwise_ Elementwise() {
    return Elementwise_(*this);
  }

  __host__ __device__ Elementwise_ el() {
    return Elementwise_(*this);
  }

  template <typename Z, std::enable_if_t<are_addble_v<data_type, Z>, bool> = true>
  __host__ __device__ inline friend auto operator/(const Z &val, const Elementwise_ &mat_) {
    // printf("%i %i\n", std::is_convertible_v<Z, U>, std::is_convertible_v<U, Z>);
    Mat_Base<TP, N, sym_tp> &mat = mat_.mat_;
    Mat_Base<decltype(mat(0, 0) + val), N, sym_tp> res;

#pragma unroll
    for (int ind = 0; ind < DATA_SZ; ind++) {
      res[ind] = val / mat[ind];
    }

    return res;
  }

  template <typename A, typename B>
  __host__ __device__ auto Elementwise(A (&f)(B)) {

    const auto &mat = *this;
    Mat_Base<decltype(f(mat(0))), N, sym_tp> res;
    // Vec< F, N> mat_out;

#pragma unroll
    for (int ind = 0; ind < DATA_SZ; ind++) {
      res[ind] = f(mat[ind]);
    }
    return res;
  }

  template <typename A, typename B>
  __host__ __device__ auto For_each(A (&f)(B)) {

    auto &mat = *this;
    // Vec< F, N> mat_out;

#pragma unroll
    for (int ind = 0; ind < DATA_SZ; ind++) {
      mat[ind] = f(mat[ind]);
    }
    return *this;
  }

  /* __host__ __device__ inline void print() const {
     const auto &a = *this;
  #pragma unroll
     for (int i = 0; i < N; i++) {
  #pragma unroll
       for (int j = 0; j < N; j++) {
         printf("%f ", value_type(a(i, j)));
       }
       printf("\n");
     }
     printf("\n");
   }*/

  template <typename U = value_type, std::enable_if_t<std::is_floating_point_v<U>, bool> = true>
  __host__ __device__ inline void print() const {
    const auto &a = *this;
#pragma unroll
    for (int i = 0; i < N; i++) {
#pragma unroll
      for (int j = 0; j < N; j++) {
        printf("%f ", float(a(i, j)));
      }
      printf("\n");
    }
    printf("\n");
  }

  template <typename U = value_type, std::enable_if_t<std::is_integral_v<U>, bool> = true>
  __host__ __device__ inline void print() const {
    const auto &a = *this;
#pragma unroll
    for (int i = 0; i < N; i++) {
#pragma unroll
      for (int j = 0; j < N; j++) {
        printf("%d ", int(a(i, j)));
      }
      printf("\n");
    }
    printf("\n");
  }

  template <typename U = value_type, std::enable_if_t<std::is_pointer_v<U>, bool> = true>
  __host__ __device__ inline void print() const {
    const auto &a = *this;
#pragma unroll
    for (int i = 0; i < N; i++) {
#pragma unroll
      for (int j = 0; j < N; j++) {
        printf("%p ", (a(i, j)));
      }
      printf("\n");
    }
    printf("\n");
  }

  friend std::ostream &operator<<(std::ostream &os, const Mat_Base<TP, N, sym_tp> mat) {

#pragma unroll
    for (int i = 0; i < N; i++) {
#pragma unroll
      for (int j = 0; j < N; j++) {
        os << mat(i, j) << " ";
      }
      os << std::endl;
    }
    return os;
  };
};

/*
  template <int N, typename TP ,typename Z, mat_sym_tp sym_tp , mat_sym_tp sym_tp_b>
  __host__ __device__ inline auto operator*(const Mat_Base<TP, N, sym_tp> &a , const Mat_Base<Z, N, sym_tp_b> &b) {
    Mat_Base<decltype(a(0, 0) * b(0, 0)), N, mat_sym_tp::none> c;

   // printf(" %f %f \n %f %f \n\n", a(0,0), a(0,1) , a(1,0) , a(1,1) );
   // printf(" %f %f \n %f %f \n\n", b(0,0), b(0,1) , b(1,0) , b(1,1) );
#pragma unroll
    for (int i = 0; i < N; i++) {
#pragma unroll
      for (int j = 0; j < N; j++) {
        c(i, j) = a(i, 0) * b(0, j);
#pragma unroll
        for (int k = 1; k < N; k++) {
          c(i, j) += a(i, k) * b(k, j);
        }
      }
    }
    return c;
  }
*/
/////////////////////////////////
////////////////////////////////

template <typename TP, int N>
class Vec {
private:
  std::array<TP, N> data_;

public:
  typedef tensor_helper::get_value_type<TP> value_type;
  typedef cuda::std::remove_pointer_t<value_type> data_type;

  auto begin() {
    return data_.begin();
  };
  auto end() {
    return data_.end();
  };

  auto cbegin() {
    return data_.cbegin();
  };
  auto cend() {
    return data_.cend();
  };

  // TP &x = data_[0], &y = data_[1], &z = data_[2];

  __host__ __device__ inline TP *Data() {
    return &data_[0];
  }

  __host__ __device__ inline const TP *Data() const {
    return &data_[0];
  }

  __host__ __device__ inline TP &operator()(int i) {
    return data_[i];
  }

  __host__ __device__ inline const TP operator()(int i) const {
    return data_[i];
  }

  __host__ __device__ inline TP &operator[](int i) {
    return data_[i];
  }

  __host__ __device__ inline const TP operator[](int i) const {
    return data_[i];
  }

  __host__ __device__ inline Vec(){};

  //__host__ __device__ inline explicit Vec(TP val) { std::fill(begin(), end(), val); };

  __host__ __device__ inline explicit Vec(TP val) {
#pragma unroll
    for (int ind = 0; ind < N; ind++) {
      (*this)(ind) = val;
    }
  };

  template <typename U>
  __host__ __device__ inline Vec(Vec<U, N> vec_in) {
#pragma unroll
    for (int ind = 0; ind < N; ind++) {
      (*this)(ind) = TP(vec_in(ind));
    }
  };

  template <typename... Ts, typename = std::enable_if_t<are_all_conv_to_T_v<TP, Ts...> and has_SZ<N, Ts...>>>
  __host__ __device__ inline Vec(Ts &&...ts) : data_{TP(ts)...} {};

  __host__ __device__ inline Vec(const std::array<TP, N> &arr) : data_(arr){};

  __host__ __device__ inline Vec(const std::initializer_list<TP> list) { // v

    auto it = list.begin();
#pragma unroll
    for (int ind = 0; ind < N; ind++, it++) {
      (*this)[ind] = *it;
    }
  };

template <int M=N, typename = std::enable_if_t< (M>1) > >
  __host__ __device__ inline Vec(const TP vec_in[N]) { // v

#pragma unroll
    for (int ind = 0; ind < N; ind++) {
      (*this)[ind] = vec_in[ind];
    }
  };

  // template <std::enable_if_t<N==3, bool> = true>
  __host__ __device__ inline Vec(dim3 dims) { // v
    static_assert(N == 3, "only Vec3 can be cast to dim3");
    auto &vec = *this;
    // printf("fuck %g %g %g\n", TP(dims.x), TP(dims.y) , TP(dims.z) );
    vec(0) = TP(dims.x);
    vec(1) = TP(dims.y);
    vec(2) = TP(dims.z);
  };

  template <std::size_t... inds>
  __host__ __device__ inline Vec<tensor_helper::ref_wrap<std::remove_pointer_t<value_type>>, N>
  d_ref_impl(std::index_sequence<inds...>) {
    Vec &vec = *this;
    return Vec<tensor_helper::ref_wrap<std::remove_pointer_t<value_type>>, N>(tensor_helper::ref_wrap(*(vec.data_[inds]))...);
  }; // v

  __host__ __device__ inline Vec<tensor_helper::ref_wrap<std::remove_pointer_t<value_type>>, N>
  operator*() {
    return d_ref_impl(std::make_index_sequence<N>{});
  }; // v

  template <std::size_t... inds>
  __host__ __device__ inline Vec<tensor_helper::ref_wrap<std::remove_pointer_t<value_type>>, N>
  d_ref_impl(std::index_sequence<inds...>) const {
    Vec &vec = *this;
    return Vec<tensor_helper::ref_wrap<std::remove_pointer_t<value_type>>, N>(tensor_helper::ref_wrap(*(vec.data_[inds]))...);
  }; // v

  __host__ __device__ inline Vec<tensor_helper::ref_wrap<std::remove_pointer_t<value_type>>, N>
  operator*() const {
    return d_ref_impl(std::make_index_sequence<N>{});
  }; // v

  // template <std::enable_if_t<N==3, bool> = true>
  __host__ __device__ inline operator dim3() { // v
    static_assert(N == 3, "only Vec3 can be cast to dim3");
    auto vec = *this;
    dim3 dims = {uint(vec(0)), uint(vec(1)), uint(vec(2))};
    return dims;
  };

  template <typename U, std::enable_if_t<are_addble_as_T_v<value_type, U>, bool> = true> // v
  __host__ __device__ inline auto &operator+=(const U &val) {
    Vec &vec = *this;
#pragma unroll
    for (int i = 0; i < N; i++) {
      vec(i) += val;
    }
    return *this;
  }

  template <typename U, std::enable_if_t<are_addble_v<value_type, U>, bool> = true> // v
  __host__ __device__ inline auto operator+(const U &val) {
    Vec &vec = *this;
    Vec<decltype(val + vec(0)), N> res;
#pragma unroll
    for (int i = 0; i < N; i++) {
      res(i) = vec(i) + val;
    }
    return res;
  }

  template <typename U, std::enable_if_t<are_addble_as_T_v<value_type, U>, bool> = true>
  __host__ __device__ inline auto &operator=(const U &val) {
    Vec &vec = *this;
#pragma unroll
    for (int i = 0; i < N; i++) {
      vec(i) = val;
    }
    return *this;
  }

  template <typename U>
  __host__ __device__ inline auto &operator=(const Vec<U, N> &rhs) { // v
    Vec &vec = *this;
#pragma unroll
    for (int i = 0; i < N; i++) {
      vec(i) = rhs(i);
    }
    return *this;
  }

  template <typename U>
  __host__ __device__ inline auto &operator+=(const Vec<U, N> &rhs) { // v
    Vec &vec = *this;
#pragma unroll
    for (int i = 0; i < N; i++) {
      vec(i) += rhs(i);
    }
    return *this;
  }

  __host__ __device__ inline auto operator-() {
    Vec &vec = *this;
    Vec<value_type, N> res;
#pragma unroll
    for (int i = 0; i < N; i++) {
      res(i) = -vec(i);
    }
    return res;
  }

  template <typename U, std::enable_if_t<are_addble_as_T_v<value_type, U>, bool> = true>
  __host__ __device__ inline auto &operator-=(const U &val) {
    Vec &vec = *this;
#pragma unroll
    for (int i = 0; i < N; i++) {
      vec(i) -= val;
    }
    return *this;
  }

  template <typename U, std::enable_if_t<are_addble_v<value_type, U>, bool> = true> // v
  __host__ __device__ inline auto operator-(const U &val) {
    Vec &vec = *this;
    Vec<decltype(val + vec(0)), N> res;
#pragma unroll
    for (int i = 0; i < N; i++) {
      res(i) = vec(i) - val;
    }
    return res;
  }

  template <typename U, std::enable_if_t<are_addble_as_T_v<value_type, U>, bool> = true> // v
  __host__ __device__ inline auto &operator-=(const Vec<U, N> &rhs) {
    Vec &vec = *this;
#pragma unroll
    for (int i = 0; i < N; i++) {
      vec(i) -= rhs(i);
    }
    return *this;
  }

  template <typename U>
  __host__ __device__ inline bool operator==(const Vec<U, N> &rhs) const { // v
    const Vec &vec = *this;

#pragma unroll
    for (int i = 0; i < N; i++) {
      if (vec(i) != rhs(i))
        return false;
    }

    return true;
  }

  template <typename U, std::enable_if_t<are_addble_as_T_v<value_type, U>, bool> = true> // v
  __host__ __device__ inline auto &operator*=(const U &d) {
    Vec &vec = *this;
#pragma unroll
    for (int i = 0; i < N; i++) {
      vec(i) *= d;
    }

    return *this;
  }

  template <typename U, std::enable_if_t<are_addble_as_T_v<value_type, U>, bool> = true> // v
  __host__ __device__ inline auto &operator/=(const U &d) {
    Vec &vec = *this;
#pragma unroll
    for (int i = 0; i < N; i++) {
      vec(i) /= d;
    }

    return *this;
  }

  /*
    template <typename U, std::enable_if_t<are_addble_v<TP, U>, bool> = true>
    __host__ __device__ inline auto operator+(const U &rhs) const {
      const Vec &vec = *this;
      Vec<decltype(vec(0) * rhs), N> res;

  #pragma unroll
      for (int i = 0; i < N; i++) {
        res(i) = vec(i) + rhs;
      }
      return res;
    } */

  /*
    template <typename U, std::enable_if_t<any_conv_to_v<TP, U>, bool> = true>
    __host__ __device__ inline auto operator-(const U &rhs) const {
      const Vec &vec = *this;
      Vec<decltype(vec(0) * rhs), N> res;

  #pragma unroll
      for (int i = 0; i < N; i++) {
        res(i) = vec(i) - rhs;
      }
      return res;
    } */

  template <typename U, std::enable_if_t<any_conv_to_v<TP, U>, bool> = true> // v
  __host__ __device__ inline auto operator*(const U &d) const {
    const Vec &vec = *this;
    Vec<decltype(vec(0) * d), N> res;

#pragma unroll
    for (int i = 0; i < N; i++) {
      res(i) = vec(i) * d;
    }
    return res;
  }

  template <typename U, std::enable_if_t<any_conv_to_v<TP, U>, bool> = true> // v
  __host__ __device__ inline auto operator/(const U &d) const {
    const Vec &vec = *this;
    Vec<decltype(vec(0) * d), N> res;

#pragma unroll
    for (int i = 0; i < N; i++) {
      res(i) = vec(i) / d;
    }
    return res;
  }

  template <typename U>
  __host__ __device__ inline auto operator+(const Vec<U, N> &rhs) const { // v
    const Vec &vec = *this;

    Vec<decltype(vec(0) + rhs(0)), N> res;
#pragma unroll
    for (int i = 0; i < N; i++) {
      res(i) = vec(i) + rhs(i);
    }
    return res;
  }

  template <typename U>
  __host__ __device__ inline auto operator-(const Vec<U, N> &rhs) const { // v
    const Vec &vec = *this;

    Vec<decltype(vec(0) + rhs(0)), N> res;
#pragma unroll
    for (int i = 0; i < N; i++) {
      res(i) = vec(i) - rhs(i);
    }
    return res;
  }

  template <typename Z>
  __host__ __device__ inline auto operator*(const Vec<Z, N> d) const { // v
    const Vec &vec = *this;

    decltype(vec(0) * d(0)) res = vec(0) * d(0);
#pragma unroll
    for (int i = 1; i < N; i++) {
      res += vec(i) * d(i);
    }
    return res;
  }

  template <typename Z, mat_sym_tp sym_tp>
  __host__ __device__ inline auto operator*(const Mat_Base<Z, N, sym_tp> &mat) const { // v
    const Vec &vec = *this;
    typedef decltype(vec(0) + mat(0, 0)) out_tp;
    Vec<out_tp, N> res;

#pragma unroll
    for (int i = 0; i < N; i++) {
      res(i) = vec(0) * mat(0, i);
#pragma unroll
      for (int j = 1; j < N; j++)
        res(i) += vec(j) * mat(j, i);
    }
    return res;
  }

  template <typename Z>
  __host__ __device__ inline auto Outer(const Vec<Z, N> &rhs) const { // v
    const Vec &vec = *this;
    Mat_Base<decltype(vec(0) + rhs(0)), N, mat_sym_tp::none> res;

#pragma unroll
    for (int i = 0; i < N; i++) {
#pragma unroll
      for (int j = 0; j < N; j++)
        res(i, j) = vec(i) * rhs(j);
    }

    return res;
  }

  __host__ __device__ inline auto Outer() const { // v
    const Vec &vec = *this;
    Mat_Base< value_type, N, mat_sym_tp::sym> res;

#pragma unroll
    for (int i = 0; i < N; i++) {
#pragma unroll
      for (int j = i; j < N; j++)
        res(i, j) = vec(i) * vec(j);
    }

    return res;
  }

  __host__ __device__ inline auto Proj_op() const { // v
    const Vec &vec = *this;
    const Vec<value_type, N> vec_2normed = vec / vec.Norm_Square();
    Mat_Base<value_type, N, mat_sym_tp::sym> res;

#pragma unroll
    for (int i = 0; i < N; i++) {
#pragma unroll
      for (int j = i; j < N; j++)
        res(i, j) = vec(i) * vec_2normed(j);
    }

    return res;
  }

  template <typename Z, int M = N, std::enable_if_t<M == 3, bool> = true> // V
  __host__ __device__ inline auto Cross(const Vec<Z, 3> &rhs) const {
    const Vec &vec = *this;
    Vec<decltype(vec(0) + rhs(0)), 3> res(vec(1) * rhs(2) - vec(2) * rhs(1),
                                          vec(2) * rhs(0) - vec(0) * rhs(2),
                                          vec(0) * rhs(1) - vec(1) * rhs(0));
    return res;
  }

  __host__ __device__ inline value_type Norm_Square() const { // V
    const Vec &vec = *this;

    value_type sum = vec(0) * vec(0);
#pragma unroll
    for (int i = 1; i < N; i++) {
      sum += vec(i) * vec(i);
    }
    return sum;
  }

  class Elementwise_ {
  public:
    Vec<TP, N> &vec_;

    __host__ __device__ Elementwise_(Vec<TP, N> &vec_in) : vec_{vec_in} {};

    template <typename Z>
    __host__ __device__ inline auto operator*=(const Vec<Z, N> &vec2) { // v

#pragma unroll
      for (int i = 0; i < N; i++) {
        vec_(i) *= vec2(i);
      }

      return vec_;
    };

    template <typename Z>
    __host__ __device__ inline auto operator*(const Vec<Z, N> &vec2) const { // v
      Vec<decltype(vec_(0) * vec2(0)), N> res;

#pragma unroll
      for (int i = 0; i < N; i++) {
        res(i) = vec_(i) * vec2(i);
      }

      return res;
    };

    template <typename Z>
    __host__ __device__ auto operator/=(const Vec<Z, N> &vec2) { // v

#pragma unroll
      for (int i = 0; i < N; i++) {
        vec_(i) /= vec2(i);
      }

      return vec_;
    };

    template <typename Z>
    __host__ __device__ Vec operator/(const Vec<Z, N> &vec2) const { // v
      Vec<decltype(vec_(0) * vec2(0)), N> res;

#pragma unroll
      for (int i = 0; i < N; i++) {
        res(i) = vec_(i) / vec2(i);
      }

      return res;
    };
  };

  __host__ __device__ Elementwise_ Elementwise() {
    return Elementwise_(*this);
  }

  template <typename Z, std::enable_if_t<are_addble_v<data_type, Z>, bool> = true> // v
  __host__ __device__ inline friend auto operator/(const Z &val, const Elementwise_ &vec_) {

    Vec<TP, N> &vec = vec_.vec_;
    Vec<decltype(vec(0) * val), N> res;

#pragma unroll
    for (int i = 0; i < N; i++) {
      res(i) = val / vec(i);
    }
    return res;
  }

  // template <typename A, typename B>
  //__host__ __device__ auto Elementwise( cuda::std::function<A(B)> const& f){
  //__host__ __device__ auto Elementwise(A (&f)(B)) {
  template <class F>
  __host__ __device__ auto Elementwise(F &&f) {
    const Vec &vec = *this;
    Vec<decltype(f(vec(0))), N> res;
    // Vec< F, N> mat_out;

#pragma unroll
    for (int ind = 0; ind < N; ind++) {
      res[ind] = f(vec[ind]);
    }
    return res;
  }

  template <class F, class G>
  __host__ __device__ auto Elementwise(Vec<G, N> vec2, F &&f) {
    const Vec &vec = *this;
    Vec<decltype(f(vec(0), vec2(0))), N> res;
    // Vec< F, N> mat_out;

#pragma unroll
    for (int ind = 0; ind < N; ind++) {
      res(ind) = f(vec(ind), vec2(ind));
    }
    return res;
  }

  template <typename... U>
  __host__ __device__ auto el(U... args) {
    return Elementwise(args...);
  }

  template <class F>
  __host__ __device__ auto Reduce(F &&f) {
    const Vec &vec = *this;
    decltype(f(vec(0), vec(0))) res = f(vec(1), vec(0));
    // Vec< F, N> mat_out;

#pragma unroll
    for (int ind = 2; ind < N; ind++) {
      res = f(vec(ind), res);
    }
    return res;
  }

  template <typename A, typename B>
  __host__ __device__ auto For_each(A (&f)(B)) {

    Vec &vec = *this;
    // Vec< F, N> mat_out;

#pragma unroll
    for (int ind = 0; ind < N; ind++) {
      vec[ind] = f(vec[ind]);
    }
    return *this;
  }

  /* __host__ __device__ inline void print() const {
     const Vec &a = *this;
  #pragma unroll
     for (int j = 0; j < N; j++) {
       printf("%f ", float(a(j)));
     }
     printf("\n");
   }*/

  template <typename U = value_type, std::enable_if_t<std::is_floating_point_v<U>, bool> = true>
  __host__ __device__ inline void print() const {
    const auto &a = *this;
#pragma unroll
    for (int i = 0; i < N; i++) {
      printf("%f ", float(a(i)));
    }
    printf("\n");
  }

  template <typename U = value_type, std::enable_if_t<std::is_integral_v<U>, bool> = true>
  __host__ __device__ inline void print() const {
    const auto &a = *this;
#pragma unroll
    for (int i = 0; i < N; i++) {
      printf("%d ", int(a(i)));
    }
    printf("\n");
  }

  template <typename U = value_type, std::enable_if_t<std::is_pointer_v<U>, bool> = true>
  __host__ __device__ inline void print() const {
    const auto &a = *this;
#pragma unroll
    for (int i = 0; i < N; i++) {
      printf("%p ", (a(i)));
    }
    printf("\n");
  }

  friend std::ostream &operator<<(std::ostream &os, const Vec<TP, N> vec) {

#pragma unroll
    for (int ind = 0; ind < N; ind++) {
      os << vec(ind) << " ";
    }
    return os;
  };
};

// template <typename Z, typename U, int N, std::enable_if_t<any_conv_to_v<U, Z>, bool> = true>  //v
template <typename Z, typename U, int N, std::enable_if_t<are_addble_v<U, Z>, bool> = true> // v
__host__ __device__ inline auto operator*(const Z &val, const Vec<U, N> &vec) {
  return vec * val;
}

// template <typename Z, typename U, int N, mat_sym_tp sym_tp, std::enable_if_t<any_conv_to_v<U, Z>, bool> = true>  //v
template <typename Z, typename U, int N, mat_sym_tp sym_tp, std::enable_if_t<any_conv_to_v<U, Z>, bool> = true>
__host__ __device__ inline auto operator*(const Z &val, const Mat_Base<U, N, sym_tp> &mat) {
  // printf("%i %i\n", std::is_convertible_v<Z, U>, std::is_convertible_v<U, Z>);
  return mat * val;
}

template <typename TP, int N, mat_sym_tp sym_tp>
template <typename Z>
__host__ __device__ inline auto Mat_Base<TP, N, sym_tp>::operator*(const Vec<Z, N> &vec) const {
  const Mat_Base &mat = *this;
  typedef decltype(vec(0) + mat(0, 0)) out_tp;
  Vec<out_tp, N> res;

#pragma unroll
  for (int i = 0; i < N; i++) {
    res(i) = mat(i, 0) * vec(0);
#pragma unroll
    for (int j = 1; j < N; j++)
      res(i) += mat(i, j) * vec(j);
  }
  return res;
}

template <typename TP, int N, mat_sym_tp sym_tp>
template <std::size_t... inds>
__host__ __device__ inline Vec<tensor_helper::ref_wrap<TP>, N>
Mat_Base<TP, N, sym_tp>::Diag_impl(std::index_sequence<inds...>) { // V
  Mat_Base &mat = *this;
  return Vec<tensor_helper::ref_wrap<TP>, N>(tensor_helper::ref_wrap(mat(inds, inds))...);
}

template <typename TP, int N, mat_sym_tp sym_tp>
__host__ __device__ inline Vec<tensor_helper::ref_wrap<TP>, N> Mat_Base<TP, N, sym_tp>::Diag() { // V
  // Mat_Base &mat = *this;
  return Diag_impl(std::make_index_sequence<N>{});
}

template <typename TP, int N, mat_sym_tp sym_tp>
template <std::size_t... inds>
__host__ __device__ inline Vec<tensor_helper::ref_wrap<TP>, N>
Mat_Base<TP, N, sym_tp>::Diag_impl(std::index_sequence<inds...>) const { // V
  Mat_Base &mat = *this;
  return Vec<tensor_helper::ref_wrap<TP>, N>(tensor_helper::ref_wrap(mat(inds, inds))...);
}

template <typename TP, int N, mat_sym_tp sym_tp>
__host__ __device__ inline Vec<tensor_helper::ref_wrap<TP>, N> Mat_Base<TP, N, sym_tp>::Diag() const { // V
  // Mat_Base &mat = *this;
  return Diag_impl(std::make_index_sequence<N>{});
};

template <typename TP, int N, mat_sym_tp sym_tp>
template <std::size_t... inds>
__host__ __device__ inline Vec<tensor_helper::ref_wrap<TP>, N> Mat_Base<TP, N, sym_tp>::Row_impl(int row_num, std::index_sequence<inds...>) { // v
  Mat_Base &mat = *this;
  return Vec<tensor_helper::ref_wrap<TP>, N>(tensor_helper::ref_wrap(mat(row_num, inds))...);
}

template <typename TP, int N, mat_sym_tp sym_tp>
__host__ __device__ inline Vec<tensor_helper::ref_wrap<TP>, N> Mat_Base<TP, N, sym_tp>::Row(int row_num) { // v
  // Mat3 &mat = *this;
  // return Vec3<tensor_helper::ref_wrap<TP>>(mat(Row_num, 0), mat(Row_num, 1),mat(Row_num, 2));
  return Row_impl(row_num, std::make_index_sequence<N>{});
}

template <typename TP, int N, mat_sym_tp sym_tp>
template <std::size_t... inds>
__host__ __device__ inline Vec<tensor_helper::ref_wrap<TP>, N> Mat_Base<TP, N, sym_tp>::Row_impl(int row_num, std::index_sequence<inds...>) const { // v
  Mat_Base &mat = *this;
  return Vec<tensor_helper::ref_wrap<TP>, N>(tensor_helper::ref_wrap(mat(row_num, inds))...);
}

template <typename TP, int N, mat_sym_tp sym_tp>
__host__ __device__ inline Vec<tensor_helper::ref_wrap<TP>, N> Mat_Base<TP, N, sym_tp>::Row(int row_num) const { // v
  // Mat3 &mat = *this;
  // return Vec3<tensor_helper::ref_wrap<TP>>(mat(Row_num, 0), mat(Row_num, 1),mat(Row_num, 2));
  return Row_impl(row_num, std::make_index_sequence<N>{});
}

template <typename TP, int N, mat_sym_tp sym_tp>
template <std::size_t... inds>
__host__ __device__ inline Vec<tensor_helper::ref_wrap<TP>, N> Mat_Base<TP, N, sym_tp>::Col_impl(int col_num, std::index_sequence<inds...>) { // v
  Mat_Base &mat = *this;
  return Vec<tensor_helper::ref_wrap<TP>, N>(tensor_helper::ref_wrap(mat(inds, col_num))...);
}

template <typename TP, int N, mat_sym_tp sym_tp>
__host__ __device__ inline Vec<tensor_helper::ref_wrap<TP>, N> Mat_Base<TP, N, sym_tp>::Col(int col_num) { // v
  // Mat3 &mat = *this;
  // return Vec3<tensor_helper::ref_wrap<TP>>(mat(Row_num, 0), mat(Row_num, 1),mat(Row_num, 2));
  return Col_impl(col_num, std::make_index_sequence<N>{});
}

template <typename TP, int N, mat_sym_tp sym_tp>
template <std::size_t... inds>
__host__ __device__ inline Vec<tensor_helper::ref_wrap<TP>, N> Mat_Base<TP, N, sym_tp>::Col_impl(int col_num, std::index_sequence<inds...>) const { // v
  Mat_Base &mat = *this;
  return Vec<tensor_helper::ref_wrap<TP>, N>(tensor_helper::ref_wrap(mat(inds, col_num))...);
}

template <typename TP, int N, mat_sym_tp sym_tp>
__host__ __device__ inline Vec<tensor_helper::ref_wrap<TP>, N> Mat_Base<TP, N, sym_tp>::Col(int col_num) const { // v
  // Mat3 &mat = *this;
  // return Vec3<tensor_helper::ref_wrap<TP>>(mat(Row_num, 0), mat(Row_num, 1),mat(Row_num, 2));
  return Col_impl(col_num, std::make_index_sequence<N>{});
}

template <class TP, int N> // this just a container for the edge
class inplane_tensor {

  Vec<Vec<TP, N>, N - 1> data_;

public:
  __host__ __device__ inline TP &operator()(const int &i, const int &j) { // V
    return data_(j)(i);
  }

  __host__ __device__ inline TP operator()(const int &i, const int &j) const { // V
    return data_(j)(i);
  }

  __host__ __device__ inline auto &operator()(const int &i) { // V
    return data_(i);
  }

  __host__ __device__ inline auto operator()(const int &i) const { // V
    return data_(i);
  }

  template <typename Z, mat_sym_tp sym_tp_b>
  __host__ __device__ inline auto operator*(const Mat_Base<Z, N - 1, sym_tp_b> &b) const { // v
    const auto &a = *this;
    Mat_Base<decltype(a(0, 0) * b(0, 0)), N, mat_sym_tp::none> c;

#pragma unroll
    for (int i = 0; i < N; i++) {
#pragma unroll
      for (int j = 0; j < N - 1; j++) {
        c(i, j) = a(i, 0) * b(0, j);
#pragma unroll
        for (int k = 1; k < N - 1; k++) {
          c(i, j) += a(i, k) * b(k, j);
        }
      }
    }

#pragma unroll
    for (int i = 0; i < N; i++) {
      c(i, N - 1) = TP(0);
    }

    return c;
  }

  __host__ __device__ inline auto Metric() const { // V

    const auto &a = *this;
    Mat_Base<decltype(a(0, 0) * a(0, 0)), N - 1, mat_sym_tp::sym> c;

#pragma unroll
    for (int i = 0; i < N - 1; i++) {
#pragma unroll
      for (int j = i; j < N - 1; j++) {

        c(i, j) = a(i) * a(j);
      }
    }

    return c;
  }
};

///////////////////////

//===
template <typename TP>
__host__ __device__ __forceinline__ void mat_adj(TP (&mat)[3][3],
                                                 TP (&adj_mat)[3][3]) {
  adj_mat[0][0] = -mat[1][2] * mat[2][1] + mat[1][1] * mat[2][2];
  adj_mat[0][1] = mat[0][2] * mat[2][1] - mat[0][1] * mat[2][2];
  adj_mat[0][2] = -mat[0][2] * mat[1][1] + mat[0][1] * mat[1][2];

  adj_mat[1][0] = mat[1][2] * mat[2][0] - mat[1][0] * mat[2][2];
  adj_mat[1][1] = -mat[0][2] * mat[2][0] + mat[0][0] * mat[2][2];
  adj_mat[1][2] = mat[0][2] * mat[1][0] - mat[0][0] * mat[1][2];

  adj_mat[2][0] = -mat[1][1] * mat[2][0] + mat[1][0] * mat[2][1];
  adj_mat[2][1] = mat[0][1] * mat[2][0] - mat[0][0] * mat[2][1];
  adj_mat[2][2] = -mat[0][1] * mat[1][0] + mat[0][0] * mat[1][1];
}

template <typename TP>
__host__ __device__ __forceinline__ void sym_mat_adj(TP (&mat)[3][3],
                                                     TP (&adj_mat)[3][3]) {
  adj_mat[0][0] = -mat[1][2] * mat[1][2] + mat[1][1] * mat[2][2];
  adj_mat[0][1] = mat[0][2] * mat[1][2] - mat[0][1] * mat[2][2];
  adj_mat[0][2] = -mat[0][2] * mat[1][1] + mat[0][1] * mat[1][2];

  adj_mat[1][1] = -mat[0][2] * mat[0][2] + mat[0][0] * mat[2][2];
  adj_mat[1][2] = mat[0][2] * mat[0][1] - mat[0][0] * mat[1][2];

  adj_mat[2][2] = -mat[0][1] * mat[0][1] + mat[0][0] * mat[1][1];

  adj_mat[1][0] = adj_mat[0][1];
  adj_mat[2][0] = adj_mat[0][2];

  adj_mat[2][1] = adj_mat[1][2];
}
//===

//===========//===========//===========//===========//===========

template <typename TP>
__host__ __device__ __forceinline__ void
Calc_C(TP (&F)[3][3], TP (&C)[3][3]) { // (F(tr)*F-I)/2

#pragma unroll
  for (int i = 0; i < 3; i++) {
#pragma unroll
    for (int j = 0; j <= i; j++) {
      C[i][j] = (F[0][i] * F[0][j] + F[1][i] * F[1][j] + F[2][i] * F[2][j]);
    }
  }

  C[0][1] = C[1][0];
  C[0][2] = C[2][0];
  C[1][2] = C[2][1];
}

template <typename TP>
__host__ __device__ __forceinline__ void
Calc_E(TP (&F)[3][3], TP (&E)[3][3]) { // (F(tr)*F-I)/2

#pragma unroll
  for (int i = 0; i < 3; i++) {
#pragma unroll
    for (int j = 0; j <= i; j++) {
      E[i][j] =
          (F[0][i] * F[0][j] + F[1][i] * F[1][j] + F[2][i] * F[2][j]) * TP(0.5);
    }
  }

  E[0][0] -= TP(0.5);
  E[1][1] -= TP(0.5);
  E[2][2] -= TP(0.5);
  E[0][1] = E[1][0];
  E[0][2] = E[2][0];
  E[1][2] = E[2][1];
}

template <typename TP>
__host__ __device__ __forceinline__ void
Calc_E_2d(TP (&F)[3][3], TP (&E)[2][2]) { // (F(tr)*F-I)/2

#pragma unroll
  for (int i = 0; i < 2; i++) {
#pragma unroll
    for (int j = 0; j <= i; j++) {
      E[i][j] =
          (F[0][i] * F[0][j] + F[1][i] * F[1][j] + F[2][i] * F[2][j]) * TP(0.5);
    }
  }

  E[0][0] -= TP(0.5);
  E[1][1] -= TP(0.5);
  E[0][1] = E[1][0];
}

//======

//===========//===========//===========//===========//===========
template <typename TP>
__device__ __forceinline__ void matrix_prod3(TP (&a)[3][3], TP (&b)[3][3],
                                             TP (&c)[3][3]) { // a*b=c

#pragma unroll
  for (int i = 0; i < 3; i++) {
#pragma unroll
    for (int j = 0; j < 3; j++) {
      c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
    }
  }
}

///////
template <typename TP>
__device__ __forceinline__ void matrix_sum3(TP (&a)[3][3], TP (&b)[3][3],
                                            TP (&c)[3][3]) { // a*b=c

#pragma unroll
  for (int i = 0; i < 3; i++) {
#pragma unroll
    for (int j = 0; j < 3; j++) {
      c[i][j] = a[i][j] + b[i][j];
    }
  }
}

///////
template <typename TP>
__device__ __forceinline__ void matrix_diff3(TP (&a)[3][3], TP (&b)[3][3],
                                             TP (&c)[3][3]) { // a*b=c

#pragma unroll
  for (int i = 0; i < 3; i++) {
#pragma unroll
    for (int j = 0; j < 3; j++) {
      c[i][j] = a[i][j] - b[i][j];
    }
  }
}

//===========//===========//===========//===========//===========
