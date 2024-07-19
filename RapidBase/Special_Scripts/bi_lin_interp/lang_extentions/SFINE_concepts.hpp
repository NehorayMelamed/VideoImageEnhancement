#pragma once

#include <functional>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <type_traits>


template <typename T, typename... Ts>
constexpr bool are_all_T_v = std::conjunction_v<std::is_same<T, Ts>...>;

////
template <typename T, typename... Ts>
constexpr bool are_all_conv_to_T_v = std::conjunction_v<std::is_convertible<Ts, T>...>;

template <typename T, typename Z>
constexpr bool any_conv_to_v = std::is_convertible_v<Z, T> or std::is_convertible_v<T, Z>;

////
template <int SZ, typename... Ts>
static inline constexpr bool has_SZ = (sizeof...(Ts)) == SZ;

/////
template <typename, typename, typename = void>
constexpr bool are_addble_v{};
 
template <typename T, typename B>
constexpr bool are_addble_v< T, B, std::void_t< decltype( std::declval<T>() + std::declval<B>() ) > > = true ;


template <typename, typename, typename = void>
constexpr bool are_addble_as_T_v{};

template <typename T, typename B>
constexpr bool are_addble_as_T_v<T,B, std::void_t< decltype( std::declval<T>() + std::declval<B>() ) > > = 
                                are_addble_v<T,B> and std::is_convertible_v< T,  decltype( std::declval<T>() + std::declval<B>() ) >;

namespace std{
template<class T>
struct is_pointer<T* __restrict__> : std::true_type {};
 
template<class T>
struct is_pointer<T* __restrict__ const> : std::true_type {};
 
template<class T>
struct is_pointer<T* __restrict__ volatile> : std::true_type {};
 
template<class T>
struct is_pointer<T* __restrict__  const volatile> : std::true_type {};


template< class T > struct remove_pointer<T* __restrict__>                {typedef T type;};
template< class T > struct remove_pointer<T* __restrict__ const>          {typedef T type;};
template< class T > struct remove_pointer<T* __restrict__ volatile>       {typedef T type;};
template< class T > struct remove_pointer<T* __restrict__ const volatile> {typedef T type;};
};
