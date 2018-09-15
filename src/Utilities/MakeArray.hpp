// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function make_array.

#pragma once

#include <array>
#include <cstddef>
#include <iterator>
#include <utility>

#include "Utilities/ForceInline.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

// Much of this is taken from
// http://stackoverflow.com/questions/1065774/initialization-of
//     -a-normal-array-with-one-default-value
// with some modifications.
namespace MakeArray_detail {
// We handle the zero size case separately below because both for size zero
// and size one arrays the index_sequence is empty.
// We use the index_sequence to be able to fill the first Size-1 (which is
// sizeof...(Is)) via constructor calls `T(args...)`. The final element is
// forwarded to avoid a possible copy if an rvalue reference is passed to
// make_array. The generic implementation handles both the make_array(T&&) case
// and the make_array(Args&&...) case below.
// The (void)Is cast is to avoid any potential trouble with overloaded comma
// operators.
template <bool SizeZero>
struct MakeArray {
  template <typename T, typename... Args, size_t... Is>
  static SPECTRE_ALWAYS_INLINE constexpr std::array<T, sizeof...(Is) + 1> apply(
      std::index_sequence<Is...> /* unused */,
      Args&&... args) noexcept(noexcept(std::array<T, sizeof...(Is) + 1>{
      {((void)Is, T(args...))..., T(std::forward<Args>(args)...)}})) {
    return {{((void)Is, T(args...))..., T(std::forward<Args>(args)...)}};
  }
};

template <>
struct MakeArray<true> {
  template <typename T, typename... Args>
  static SPECTRE_ALWAYS_INLINE constexpr std::array<T, 0> apply(
      std::index_sequence<> /* unused */, Args&&... args) noexcept {
#ifndef HAVE_BROKEN_ARRAY0
    expand_pack(args...);  // Used in other preprocessor branch
    return std::array<T, 0>{{}};
#else  // HAVE_BROKEN_ARRAY0
    // https://bugs.llvm.org/show_bug.cgi?id=35491
    return {{T(std::forward<Args>(args)...)}};
#endif  // HAVE_BROKEN_ARRAY0
  }
};
}  // namespace MakeArray_detail

/*!
 * \ingroup UtilitiesGroup
 * \brief Create a `std::array<T, Size>{{T(args...), T(args...), ...}}`
 * \tparam Size the size of the array
 * \tparam T the type of the element in the array
 */
template <size_t Size, typename T, typename... Args>
SPECTRE_ALWAYS_INLINE constexpr std::array<T, Size>
make_array(Args&&... args) noexcept(
    noexcept(MakeArray_detail::MakeArray<Size == 0>::template apply<T>(
        std::make_index_sequence<(Size == 0 ? Size : Size - 1)>{},
        std::forward<Args>(args)...))) {
  return MakeArray_detail::MakeArray<Size == 0>::template apply<T>(
      std::make_index_sequence<(Size == 0 ? Size : Size - 1)>{},
      std::forward<Args>(args)...);
}

/*!
 * \ingroup UtilitiesGroup
 * \brief Create a `std::array<std::decay_t<T>, Size>{{t, t, ...}}`
 * \tparam Size the size of the array
 */
template <size_t Size, typename T>
SPECTRE_ALWAYS_INLINE constexpr auto make_array(T&& t) noexcept(noexcept(
    MakeArray_detail::MakeArray<Size == 0>::template apply<std::decay_t<T>>(
        std::make_index_sequence<(Size == 0 ? Size : Size - 1)>{},
        std::forward<T>(t)))) -> std::array<std::decay_t<T>, Size> {
  return MakeArray_detail::MakeArray<Size == 0>::template apply<
      std::decay_t<T>>(
      std::make_index_sequence<(Size == 0 ? Size : Size - 1)>{},
      std::forward<T>(t));
}

/*!
 * \ingroup UtilitiesGroup
 * \brief Helper function to initialize a std::array with varying number of
 * arguments
 */
template <typename T, typename... V, Requires<(sizeof...(V) > 0)> = nullptr>
SPECTRE_ALWAYS_INLINE constexpr auto make_array(T&& t, V&&... values) noexcept(
    noexcept(std::array<std::decay_t<T>, sizeof...(V) + 1>{
        {std::forward<T>(t), std::forward<V>(values)...}}))
    -> std::array<typename std::decay_t<T>, sizeof...(V) + 1> {
  static_assert(
      tmpl2::flat_all_v<cpp17::is_same_v<std::decay_t<T>, std::decay_t<V>>...>,
      "all types to make_array(...) must be the same");
  return std::array<std::decay_t<T>, sizeof...(V) + 1>{
      {std::forward<T>(t), std::forward<V>(values)...}};
}

namespace MakeArray_detail {
template <typename Seq, typename T,
          Requires<cpp17::is_rvalue_reference_v<Seq>> = nullptr>
constexpr T&& forward_element(T& t) noexcept {
  return std::move(t);
}
template <typename Seq, typename T,
          Requires<not cpp17::is_rvalue_reference_v<Seq>> = nullptr>
constexpr T& forward_element(T& t) noexcept {
  return t;
}

template <typename T, size_t size, typename Seq, size_t... indexes>
constexpr std::array<T, size> make_array_from_iterator_impl(
    Seq&& s,
    std::integer_sequence<
        size_t, indexes...> /*meta*/) noexcept(noexcept(std::array<T, size>{
    {forward_element<decltype(s)>(*(std::begin(s) + indexes))...}})) {
  // clang-tidy: do not use pointer arithmetic
  return std::array<T, size>{
      {forward_element<decltype(s)>(*(std::begin(s) + indexes))...}};  // NOLINT
}
}  // namespace MakeArray_detail

/*!
 * \ingroup UtilitiesGroup
 * \brief Create an `std::array<T, size>` from the first `size` values of `seq`
 *
 * \requires `Seq` has a `begin` function
 * \tparam T the type held by the array
 * \tparam size the size of the created array
 */
template <typename T, size_t size, typename Seq>
constexpr std::array<T, size> make_array(Seq&& seq) noexcept(
    noexcept(MakeArray_detail::make_array_from_iterator_impl<T, size>(
        std::forward<Seq>(seq), std::make_index_sequence<size>{}))) {
  return MakeArray_detail::make_array_from_iterator_impl<T, size>(
      std::forward<Seq>(seq), std::make_index_sequence<size>{});
}
