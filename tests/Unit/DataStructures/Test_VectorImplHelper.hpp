// Distributed under the MIT License.
// See LICENSE.txt for details.
#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/Tuple.hpp"

namespace TestHelpers {
namespace VectorImpl {
// @{
/*!
 * \brief Perform a function over combinatoric subsets of a `std::tuple`
 *
 * \details
 * If given a function object, the apply_tuple_combinations function will
 * apply the function to all ordered combinations. For instance, if the tuple
 * contains elements (a,b,c) and the function f accepts two arguments it will be
 * executed on all nine combinations of the three possible
 * arguments. Importantly, the function f must then be invokable for all such
 * argument combinations.
 *
 * \tparam ArgLength the number of arguments taken by the invokable. In the
 * case where `function` is an object or closure whose call operator is
 * overloaded or a template it is not possible to determine the number of
 * arguments using a type trait. One common use case would be a generic
 * lambda. For this reason it is necessary to explicitly specify the number of
 * arguments to the invokable.
 *
 * \param set - the tuple for which you'd like to execute every combination
 *
 * \param function - The a callable, which is called on the combinations of the
 * tuple
 *
 * \param args - a set of partially assembled arguments passed recursively. Some
 * arguments can be passed in by the using code, in which case those are kept at
 * the end of the argument list and the combinations are prependend.
 */
template <size_t ArgLength, typename T, typename... Elements, typename... Args,
          Requires<sizeof...(Args) == ArgLength> = nullptr>
constexpr inline void apply_tuple_combinations(
    const std::tuple<Elements...>& set, const T& function,
    const Args&... args) noexcept {
  (void)
      set;  // so it is used, can be documented, and causes no compiler warnings
  function(args...);
}

template <size_t ArgLength, typename T, typename... Elements, typename... Args,
          Requires<(ArgLength > sizeof...(Args))> = nullptr>
constexpr inline void apply_tuple_combinations(
    const std::tuple<Elements...>& set, const T& function,
    const Args&... args) noexcept {
  tuple_fold(set, [&function, &set, &args...](const auto& x) noexcept {
      apply_tuple_combinations<ArgLength>(set, function, x, args...);
  });
}
// @}
}  // namespace VectorImpl
}  // namespace TestHelpers
