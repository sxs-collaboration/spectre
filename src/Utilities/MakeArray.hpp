// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function make_array.

#pragma once

#include <array>

#include "ErrorHandling/Assert.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

// Much of this is taken from
// http://stackoverflow.com/questions/1065774/initialization-of
//     -a-normal-array-with-one-default-value
// with some modifications.
namespace MakeArray_detail {
/// We take integer_sequence<indexes...> as a parameter to overload the
/// function so that there is one for each type T, and one for each number of
/// indexes. To silence compiler warnings we do a static_cast<void>.
/// Use the comma operator to expand the parameter pack, then copy in value
/// (size-1) times. We use std::forward to try and move the last value into
/// place, rather than undergoing another copy.
/// Order of evaluation is well-defined for aggregate initialization, so there
/// is no risk of copy-after-move
template <std::size_t size, typename T, std::size_t... indexes>
SPECTRE_ALWAYS_INLINE constexpr std::array<std::decay_t<T>, size>
    // clang-format off
make_array_impl(
    T&& value, std::integer_sequence<size_t, indexes...> /* unused */)
    noexcept(noexcept(std::array<std::decay_t<T>, size>{
    {(static_cast<void>(indexes), value)..., std::forward<T>(value)}})) {
  return std::array<std::decay_t<T>, size>{
      {(static_cast<void>(indexes), value)..., std::forward<T>(value)}};
}
// clang-format on
}  // namespace MakeArray_detail

/// \cond HIDDEN_SYMBOLS
/// Construct empty array specialization
template <typename T>
SPECTRE_ALWAYS_INLINE constexpr std::array<std::decay_t<T>, 0> make_array(
    std::integral_constant<std::size_t, 0> /* unused */,
    T&& /* unused */) noexcept {
  return std::array<std::decay_t<T>, 0>{{}};
}
/// \endcond

/// The std::integral_constant is used to overload the function so that we call
/// the one for the correct size of the array that we want. This is done as a
/// helper function to make make_array<size_t> a simple call for users.
template <std::size_t size, typename T>
SPECTRE_ALWAYS_INLINE constexpr std::array<std::decay_t<T>, size> make_array(
    // clang-format off
    std::integral_constant<std::size_t, size> /* unused */, T&& value)
    noexcept(noexcept(MakeArray_detail::make_array_impl<size>(
    std::forward<T>(value), std::make_index_sequence<size - 1>{}))) {
  return MakeArray_detail::make_array_impl<size>(
      std::forward<T>(value), std::make_index_sequence<size - 1>{});
}
// clang-format on
/*!
 * \ingroup Utilities
 * \brief Helper class to initialize a std::array.
 *
 * \tparam size the length of the array
 */
template <std::size_t size, typename T>
SPECTRE_ALWAYS_INLINE constexpr std::array<std::decay_t<T>, size>
make_array(T&& value) noexcept(noexcept(make_array(
    std::integral_constant<std::size_t, size>{}, std::forward<T>(value)))) {
  return make_array(std::integral_constant<std::size_t, size>{},
                    std::forward<T>(value));
}

/*!
 * \ingroup Utilities
 * \brief Helper function to initialize a std::array with varying number of
 * arguments
 */
template <typename T, typename... V, Requires<(sizeof...(V) > 0)> = nullptr>
SPECTRE_ALWAYS_INLINE constexpr std::array<typename std::decay_t<T>,
                                           sizeof...(V) + 1>
make_array(T&& t, V&&... values) noexcept(
    noexcept(std::array<std::decay_t<T>, sizeof...(V) + 1>{
        {std::forward<T>(t), std::forward<V>(values)...}})) {
  static_assert(
      tmpl2::flat_all_v<cpp17::is_same_v<std::decay_t<T>, std::decay_t<V>>...>,
      "all types to make_array(...) must be the same");
  return std::array<std::decay_t<T>, sizeof...(V) + 1>{
      {std::forward<T>(t), std::forward<V>(values)...}};
}

namespace MakeArray_detail {
template <typename T, size_t size, typename Seq, size_t... indexes>
constexpr std::array<T, size> make_array_from_iterator_impl(
    Seq&& s,
    std::integer_sequence<
        size_t, indexes...> /*meta*/) noexcept(noexcept(std::array<T, size>{
    {static_cast<T>(*(std::begin(s) + indexes))...}})) {
  // clang-tidy: do not use pointer arithmetic
  return std::array<T, size>{
      {static_cast<T>(*(std::begin(s) + indexes))...}};  // NOLINT
}
}  // namespace MakeArray_detail

/*!
 * \ingroup Utilities
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
