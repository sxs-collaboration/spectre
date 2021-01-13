// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Utilities/TypeTraits/CreateHasTypeAlias.hpp"

namespace domain::BoundaryConditions {
CREATE_HAS_TYPE_ALIAS(boundary_conditions_base)
CREATE_HAS_TYPE_ALIAS_V(boundary_conditions_base)

namespace detail {
// used as passive error message
struct TheSystemHasNoBoundaryConditionsBaseTypeAlias {};

template <typename T, typename = std::void_t<>>
struct get_boundary_conditions_base {
  using type = TheSystemHasNoBoundaryConditionsBaseTypeAlias;
};

template <typename T>
struct get_boundary_conditions_base<
    T, std::void_t<typename T::boundary_conditions_base>> {
  using type = typename T::boundary_conditions_base;
};
}  // namespace detail

/// Returns `T::boundary_condition_base` or a placeholder class.
template <typename T>
using get_boundary_conditions_base =
    typename detail::get_boundary_conditions_base<T>::type;
}  // namespace domain::BoundaryConditions
