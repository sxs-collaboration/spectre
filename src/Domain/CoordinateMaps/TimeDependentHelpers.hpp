// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Utilities/TypeTraits/CreateIsCallable.hpp"
#include "Utilities/TypeTraits/IsCallable.hpp"

namespace domain {
/// Check if the calls to the coordinate map and its inverse map are
/// time-dependent
template <typename T>
using is_map_time_dependent_t = tt::is_callable_t<
    T, std::array<std::decay_t<T>, std::decay_t<T>::dim>, double,
    std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>>;

/// Check if the calls to the coordinate map and its inverse map are
/// time-dependent
template <typename T>
constexpr bool is_map_time_dependent_v = is_map_time_dependent_t<T>::value;

namespace detail {
CREATE_IS_CALLABLE(jacobian)
}  // namespace detail

/// Check if the calls to the Jacobian and inverse Jacobian of the coordinate
/// map are time-dependent
template <typename Map, typename T>
using is_jacobian_time_dependent_t = detail::is_jacobian_callable_t<
    Map, std::array<std::decay_t<T>, std::decay_t<Map>::dim>, double,
    std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>>;

/// Check if the calls to the Jacobian and inverse Jacobian of the coordinate
/// map are time-dependent
template <typename Map, typename T>
constexpr bool is_jacobian_time_dependent_v =
    is_jacobian_time_dependent_t<Map, T>::value;
}  // namespace domain
