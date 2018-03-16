// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <initializer_list>
#include <tuple>
#include <type_traits>
#include <unistd.h>  // needed on macOS for ssize_t
#include <utility>

#include "Utilities/ForceInline.hpp"

/*!
 * \ingroup UtilitiesGroup
 * \brief Specify the lower and upper bounds in a for_constexpr loop
 *
 * \see for_constexpr for_symm_lower for_symm_upper
 */
template <size_t Lower, size_t Upper>
struct for_bounds {
  static constexpr const size_t lower = Lower;
  static constexpr const size_t upper = Upper;
};

/*!
 * \ingroup UtilitiesGroup
 * \brief Specify the loop index to symmetrize over, the lower bound, and an
 * offset to add to `Index`'s current loop value in a for_constexpr loop. Loops
 * from `Lower` to `Index`'s current loop value plus `Offset`.
 *
 * \see for_constexpr for_bounds for_symm_upper
 */
template <size_t Index, size_t Lower, ssize_t Offset = 0>
struct for_symm_lower {};

/*!
 * \ingroup UtilitiesGroup
 * \brief Specify the loop index to symmetrize over and upper bounds in a
 * for_constexpr loop. Loops from the `Index`'s current loop value to `Upper`.
 *
 * \see for_constexpr for_bounds for_symm_lower
 */
template <size_t Index, size_t Upper>
struct for_symm_upper {};

namespace for_constexpr_detail {
// Provided for implementation to be self-contained
template <bool...>
struct bool_pack;
template <bool... Bs>
using all_true = std::is_same<bool_pack<Bs..., true>, bool_pack<true, Bs...>>;

// Base case
template <size_t Lower, size_t... Is, class F, class... IntegralConstants>
SPECTRE_ALWAYS_INLINE constexpr void for_constexpr_impl(
    F&& f, std::index_sequence<Is...> /*meta*/, IntegralConstants&&... v) {
  (void)std::initializer_list<char>{
      ((void)f(std::forward<IntegralConstants>(v)...,
               std::integral_constant<size_t, Is + Lower>{}),
       '0')...};
}

// Cases of second last loop
template <size_t Lower, size_t BoundsNextLower, size_t BoundsNextUpper,
          size_t... Is, class F, class... IntegralConstants>
SPECTRE_ALWAYS_INLINE constexpr void for_constexpr_impl(
    F&& f, for_bounds<BoundsNextLower, BoundsNextUpper> /*meta*/,
    std::index_sequence<Is...> /*meta*/, IntegralConstants&&... v) {
  static_assert(static_cast<ssize_t>(BoundsNextUpper) -
                        static_cast<ssize_t>(BoundsNextLower) >=
                    0,
                "Cannot make index_sequence of negative size. The upper bound "
                "in for_bounds is smaller than the lower bound.");
  (void)std::initializer_list<char>{
      ((void)for_constexpr_impl<BoundsNextLower>(
           std::forward<F>(f),
           std::make_index_sequence<(  // Safeguard against generating
                                       // index_sequence of size ~ max size_t
               static_cast<ssize_t>(BoundsNextUpper) -
                           static_cast<ssize_t>(BoundsNextLower) <
                       0
                   ? 1
                   : BoundsNextUpper - BoundsNextLower)>{},
           std::forward<IntegralConstants>(v)...,
           std::integral_constant<size_t, Is + Lower>{}),
       '0')...};
}

template <size_t Lower, size_t BoundsNextIndex, size_t BoundsNextUpper,
          size_t... Is, class F, class... IntegralConstants>
SPECTRE_ALWAYS_INLINE constexpr void for_constexpr_impl(
    F&& f, for_symm_upper<BoundsNextIndex, BoundsNextUpper> /*meta*/,
    std::index_sequence<Is...> /*meta*/, IntegralConstants&&... v) {
  static_assert(all_true<(static_cast<ssize_t>(BoundsNextUpper) -
                              static_cast<ssize_t>(std::get<BoundsNextIndex>(
                                  std::make_tuple(IntegralConstants::value...,
                                                  Is + Lower))) >=
                          0)...>::value,
                "Cannot make index_sequence of negative size. You specified an "
                "upper bound in for_symm_upper that is less than the "
                "smallest lower bound in the loop being symmetrized over.");
  (void)std::initializer_list<char>{
      ((void)for_constexpr_impl<std::get<BoundsNextIndex>(
           std::make_tuple(IntegralConstants::value..., Is + Lower))>(
           std::forward<F>(f),
           std::make_index_sequence<(  // Safeguard against generating
                                       // index_sequence of size ~ max size_t
               static_cast<ssize_t>(BoundsNextUpper) -
                           static_cast<ssize_t>(
                               std::get<BoundsNextIndex>(std::make_tuple(
                                   IntegralConstants::value..., Is + Lower))) <
                       0
                   ? 1
                   : BoundsNextUpper -
                         std::get<BoundsNextIndex>(std::make_tuple(
                             IntegralConstants::value..., Is + Lower)))>{},
           std::forward<IntegralConstants>(v)...,
           std::integral_constant<size_t, Is + Lower>{}),
       '0')...};
}

template <size_t Lower, size_t BoundsNextIndex, size_t BoundsNextLower,
          ssize_t BoundsNextOffset, size_t... Is, class F,
          class... IntegralConstants>
SPECTRE_ALWAYS_INLINE constexpr void for_constexpr_impl(
    F&& f,
    for_symm_lower<BoundsNextIndex, BoundsNextLower, BoundsNextOffset> /*meta*/,
    std::index_sequence<Is...> /*meta*/, IntegralConstants&&... v) {
  static_assert(
      all_true<(static_cast<ssize_t>(std::get<BoundsNextIndex>(
                    std::make_tuple(IntegralConstants::value..., Is + Lower))) +
                    BoundsNextOffset - static_cast<ssize_t>(BoundsNextLower) >=
                0)...>::value,
      "Cannot make index_sequence of negative size. You specified a lower "
      "bound in for_symm_lower that is larger than the upper bounds of "
      "the loop being symmetrized over");
  (void)std::initializer_list<char>{
      ((void)for_constexpr_impl<BoundsNextLower>(
           std::forward<F>(f),
           std::make_index_sequence<(  // Safeguard against generating
                                       // index_sequence of size ~ max size_t
               static_cast<ssize_t>(std::get<BoundsNextIndex>(
                   std::make_tuple(IntegralConstants::value..., Is + Lower))) +
                           BoundsNextOffset -
                           static_cast<ssize_t>(BoundsNextLower) <
                       0
                   ? 1
                   : static_cast<size_t>(
                         static_cast<ssize_t>(
                             std::get<BoundsNextIndex>(std::make_tuple(
                                 IntegralConstants::value..., Is + Lower))) +
                         BoundsNextOffset -
                         static_cast<ssize_t>(BoundsNextLower)))>{},
           std::forward<IntegralConstants>(v)...,
           std::integral_constant<size_t, Is + Lower>{}),
       '0')...};
}

// Handle cases of more than two nested loops
template <size_t Lower, class Bounds1, class... Bounds, size_t BoundsNextLower,
          size_t BoundsNextUpper, size_t... Is, class F,
          class... IntegralConstants>
SPECTRE_ALWAYS_INLINE constexpr void for_constexpr_impl(
    F&& f, for_bounds<BoundsNextLower, BoundsNextUpper> /*meta*/,
    std::index_sequence<Is...> /*meta*/, IntegralConstants&&... v) {
  static_assert(static_cast<ssize_t>(BoundsNextUpper) -
                        static_cast<ssize_t>(BoundsNextLower) >=
                    0,
                "Cannot make index_sequence of negative size. The upper bound "
                "in for_bounds is smaller than the lower bound.");

  (void)std::initializer_list<char>{
      ((void)for_constexpr_impl<BoundsNextLower, Bounds...>(
           std::forward<F>(f), Bounds1{},
           std::make_index_sequence<(  // Safeguard against generating
                                       // index_sequence of size ~ max size_t
               static_cast<ssize_t>(BoundsNextUpper) -
                           static_cast<ssize_t>(BoundsNextLower) <
                       0
                   ? 1
                   : BoundsNextUpper - BoundsNextLower)>{},
           std::forward<IntegralConstants>(v)...,
           std::integral_constant<size_t, Is + Lower>{}),
       '0')...};
}

template <size_t Lower, class Bounds1, class... Bounds, size_t BoundsNextIndex,
          size_t BoundsNextUpper, size_t... Is, class F,
          class... IntegralConstants>
SPECTRE_ALWAYS_INLINE constexpr void for_constexpr_impl(
    F&& f, for_symm_upper<BoundsNextIndex, BoundsNextUpper> /*meta*/,
    std::index_sequence<Is...> /*meta*/, IntegralConstants&&... v) {
  static_assert(all_true<(static_cast<ssize_t>(BoundsNextUpper) -
                              static_cast<ssize_t>(std::get<BoundsNextIndex>(
                                  std::make_tuple(IntegralConstants::value...,
                                                  Is + Lower))) >=
                          0)...>::value,
                "Cannot make index_sequence of negative size. You specified an "
                "upper bound in for_symm_upper that is less than the "
                "smallest lower bound in the loop being symmetrized over.");
  (void)std::initializer_list<char>{
      ((void)for_constexpr_impl<std::get<BoundsNextIndex>(
           std::make_tuple(IntegralConstants::value..., Is + Lower))>(
           std::forward<F>(f), Bounds1{},
           std::make_index_sequence<(  // Safeguard against generating
                                       // index_sequence of size ~ max size_t
               static_cast<ssize_t>(BoundsNextUpper) -
                           static_cast<ssize_t>(
                               std::get<BoundsNextIndex>(std::make_tuple(
                                   IntegralConstants::value..., Is + Lower))) <
                       0
                   ? 1
                   : BoundsNextUpper -
                         std::get<BoundsNextIndex>(std::make_tuple(
                             IntegralConstants::value..., Is + Lower)))>{},
           std::forward<IntegralConstants>(v)...,
           std::integral_constant<size_t, Is + Lower>{}),
       '0')...};
}

template <size_t Lower, class Bounds1, class... Bounds, size_t BoundsNextIndex,
          size_t BoundsNextLower, ssize_t BoundsNextOffset, size_t... Is,
          class F, class... IntegralConstants>
SPECTRE_ALWAYS_INLINE constexpr void for_constexpr_impl(
    F&& f,
    for_symm_lower<BoundsNextIndex, BoundsNextLower, BoundsNextOffset> /*meta*/,
    std::index_sequence<Is...> /*meta*/, IntegralConstants&&... v) {
  static_assert(
      all_true<(static_cast<ssize_t>(std::get<BoundsNextIndex>(
                    std::make_tuple(IntegralConstants::value..., Is + Lower))) +
                    BoundsNextOffset - static_cast<ssize_t>(BoundsNextLower) >=
                0)...>::value,
      "Cannot make index_sequence of negative size. You specified a lower "
      "bound in for_symm_lower that is larger than the upper bounds of "
      "the loop being symmetrized over");
  (void)std::initializer_list<char>{
      ((void)for_constexpr_impl<BoundsNextLower>(
           std::forward<F>(f), Bounds1{},
           std::make_index_sequence<(  // Safeguard against generating
                                       // index_sequence of size ~ max size_t
               static_cast<ssize_t>(std::get<BoundsNextIndex>(
                   std::make_tuple(IntegralConstants::value..., Is + Lower))) +
                           BoundsNextOffset -
                           static_cast<ssize_t>(BoundsNextLower) <
                       0
                   ? 1
                   : static_cast<size_t>(
                         static_cast<ssize_t>(
                             std::get<BoundsNextIndex>(std::make_tuple(
                                 IntegralConstants::value..., Is + Lower))) +
                         BoundsNextOffset -
                         static_cast<ssize_t>(BoundsNextLower)))>{},
           std::forward<IntegralConstants>(v)...,
           std::integral_constant<size_t, Is + Lower>{}),
       '0')...};
}
}  // namespace for_constexpr_detail

/// \cond
template <class Bounds0, class F>
SPECTRE_ALWAYS_INLINE constexpr void for_constexpr(F&& f) {
  static_assert(static_cast<ssize_t>(Bounds0::upper) -
                        static_cast<ssize_t>(Bounds0::lower) >=
                    0,
                "Cannot make index_sequence of negative size. The upper bound "
                "in for_bounds is smaller than the lower bound.");
  for_constexpr_detail::for_constexpr_impl<Bounds0::lower>(
      std::forward<F>(f), std::make_index_sequence<(
                              static_cast<ssize_t>(Bounds0::upper) -
                                          static_cast<ssize_t>(Bounds0::lower) <
                                      0
                                  ? 1
                                  : Bounds0::upper - Bounds0::lower)>{});
}
/// \endcond

/*!
 * \ingroup UtilitiesGroup
 * \brief Allows nested constexpr for loops including symmetrizing over loops.
 *
 * \note All upper bounds are exclusive (the comparator is a strict less than,
 * `<`)
 * \note You must loop over non-negative numbers.
 *
 * The `for_bounds` class is used for specifying the lower (first template
 * parameter) and upper (second template parameter) bounds for a single loop.
 * Symmetrizing over loops is supported via the `for_symm_lower` and
 * `for_symm_upper` classes. `for_symm_lower` takes two template parameters (and
 * a third optional one): the integer corresponding to the outer loop to
 * symmetrize over, the lower bound of the loop, and optionally an integer to
 * add to the upper bound obtained from the loop being symmetrized over.
 * This is equivalent to loops of the form:
 *
 * \code
 * for (size_t i = 0; i < Dim; ++i) {
 *   for (size_t j = param1; j < i + param2; ++j) {
 *   }
 * }
 * \endcode
 *
 * `for_symm_upper` takes two template parameters. The first is the integer
 * corresponding to the outer loop being symmetrized over, and the second is the
 * upper bound. This is equivalent to loops of the form
 *
 * \code
 * for (size_t i = 0; i < Dim; ++i) {
 *   for (size_t j = j; j < param1; ++j) {
 *   }
 * }
 * \endcode
 *
 * \example
 * Here are various example use cases of different loop structures. The runtime
 * for loops are shown for comparison. Only the elements that are 1 were mutated
 * by the `for_constexpr`.
 *
 * #### Single loops
 * Single loops are hopefully straightforward.
 * \snippet Test_ForConstexpr.cpp single_loop
 *
 * #### Double loops
 * For double loops we should the four different options for loop symmetries.
 * \snippet Test_ForConstexpr.cpp double_loop
 * \snippet Test_ForConstexpr.cpp double_symm_lower_inclusive
 * \snippet Test_ForConstexpr.cpp double_symm_lower_exclusive
 * \snippet Test_ForConstexpr.cpp double_symm_upper
 *
 * #### Triple loops
 * For triple loops we only show the double symmetrized loops, since the others
 * are very similar to the double loop case.
 * \snippet Test_ForConstexpr.cpp triple_symm_lower_lower
 * \snippet Test_ForConstexpr.cpp triple_symm_upper_lower
 *
 * \see for_bounds for_symm_lower for_symm_upper
 */
template <class Bounds0, class Bounds1, class... Bounds, class F>
SPECTRE_ALWAYS_INLINE constexpr void for_constexpr(F&& f) {
  static_assert(static_cast<ssize_t>(Bounds0::upper) -
                        static_cast<ssize_t>(Bounds0::lower) >=
                    0,
                "Cannot make index_sequence of negative size. The upper bound "
                "in for_bounds is smaller than the lower bound.");
  for_constexpr_detail::for_constexpr_impl<Bounds0::lower, Bounds...>(
      std::forward<F>(f), Bounds1{},
      std::make_index_sequence<(
          static_cast<ssize_t>(Bounds0::upper) -
                      static_cast<ssize_t>(Bounds0::lower) <
                  0
              ? 1
              : Bounds0::upper - Bounds0::lower)>{});
}
