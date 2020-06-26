// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions for slicing tuples

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

namespace tuple_impl_detail {
template <size_t Start, typename Tuple, size_t... Is>
constexpr std::tuple<std::tuple_element_t<Is + Start, std::decay_t<Tuple>>...>
slice_impl(Tuple&& tuple, std::index_sequence<Is...> /*meta*/) noexcept {
  return {std::get<Is + Start>(std::forward<Tuple>(tuple))...};
}
}  // namespace tuple_impl_detail

/// \ingroup UtilitiesGroup
/// The subset of elements in `tuple` from index `Start` to (excluding) `Stop`
template <size_t Start, size_t Stop, typename Tuple>
constexpr auto tuple_slice(Tuple&& tuple) noexcept {
  constexpr size_t tuple_size = std::tuple_size<std::decay_t<Tuple>>::value;
  static_assert(
      Start <= Stop and Stop <= tuple_size,
      "'Start' and 'Stop' must satisfy Start <= Stop <= size of 'Tuple'");
  return tuple_impl_detail::slice_impl<Start>(
      std::forward<Tuple>(tuple), std::make_index_sequence<Stop - Start>{});
}

/// \ingroup UtilitiesGroup
/// The first `Size` elements in `tuple`
template <size_t Size, typename Tuple>
constexpr auto tuple_head(Tuple&& tuple) noexcept {
  constexpr size_t tuple_size = std::tuple_size<std::decay_t<Tuple>>::value;
  static_assert(Size <= tuple_size,
                "'Size' must not exceed the size of 'Tuple'");
  return tuple_slice<0, Size>(std::forward<Tuple>(tuple));
}

/// \ingroup UtilitiesGroup
/// The last `Size` elements in `tuple`
template <size_t Size, typename Tuple>
constexpr auto tuple_tail(Tuple&& tuple) noexcept {
  constexpr size_t tuple_size = std::tuple_size<std::decay_t<Tuple>>::value;
  static_assert(Size <= tuple_size,
                "'Size' must not exceed the size of 'Tuple'");
  return tuple_slice<tuple_size - Size, tuple_size>(std::forward<Tuple>(tuple));
}
