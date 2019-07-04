// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Utilities/TypeTraits.hpp"

namespace evolution {
// @{
/// Inherits from `std::true_type` if `typename T::analytic_solution` is
/// well-formed.
template <class T, class = cpp17::void_t<>>
struct has_analytic_solution_alias : std::false_type {};
/// \cond
template <class T>
struct has_analytic_solution_alias<T,
                                   cpp17::void_t<typename T::analytic_solution>>
    : std::true_type {};
/// \endcond
template <class T>
constexpr bool has_analytic_solution_alias_v =
    has_analytic_solution_alias<T>::value;
// @}
}  // namespace evolution
