//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_ARG_H
#define IL_ARG_H

#include <tuple>
#include <utility>

namespace il {

template <std::size_t... Ints>
struct index_sequence {
  using type = index_sequence;
  using value_type = std::size_t;
  static constexpr std::size_t size() noexcept { return sizeof...(Ints); }
};

template <typename Sequence0, typename Sequence1>
struct _merge_and_renumber;

template <std::size_t... I0, std::size_t... I1>
struct _merge_and_renumber<index_sequence<I0...>, index_sequence<I1...>>
    : index_sequence<I0..., (sizeof...(I0) + I1)...> {};

template <std::size_t n>
struct make_index_sequence
    : _merge_and_renumber<typename make_index_sequence<n / 2>::type,
                          typename make_index_sequence<n - n / 2>::type> {};

template <>
struct make_index_sequence<0> : index_sequence<> {};

template <>
struct make_index_sequence<1> : index_sequence<0> {};

namespace detail {

template <typename T, typename Tuple, std::size_t... I>
constexpr T make_from_tuple_impl(Tuple&& t, index_sequence<I...>) {
  return T(std::get<I>(std::forward<Tuple>(t))...);
};

template <typename T, typename Tuple, std::size_t... I>
void placement_from_tuple_impl(T* p, Tuple&& t, index_sequence<I...>) {
  new (p) T(std::get<I>(std::forward<Tuple>(t))...);
};
}

template <typename T, typename Tuple>
constexpr T make_from_tuple(Tuple&& t) {
  return detail::make_from_tuple_impl<T>(
      std::forward<Tuple>(t),
      make_index_sequence<
          std::tuple_size<typename std::decay<Tuple>::type>::value>{});
};

template <typename T, typename Tuple>
void placement_from_tuple(T* p, Tuple&& t) {
  detail::placement_from_tuple_impl<T>(
      p, std::forward<Tuple>(t),
      make_index_sequence<
          std::tuple_size<typename std::decay<Tuple>::type>::value>{});
};

template <typename... Types>
constexpr std::tuple<Types&&...> arg(Types&&... argument) {
  return std::tuple<Types&&...>(std::forward<Types>(argument)...);
};
}

#endif  // IL_ARG_H
