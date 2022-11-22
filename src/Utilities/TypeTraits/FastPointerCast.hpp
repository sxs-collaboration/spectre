// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>
#include <utility>

#include "Utilities/Requires.hpp"

namespace tt {
namespace detail {
template <typename T, typename U, typename = void>
struct can_static_cast : std::false_type {};

template <typename T, typename U>
struct can_static_cast<T, U,
                       std::void_t<decltype(static_cast<U>(std::declval<T>()))>>
    : std::true_type {};

template <typename T, typename U>
static constexpr bool can_static_cast_v = can_static_cast<T, U>::value;
}  // namespace detail

/*!
 * \brief Cast `t` which is of type `T*` to a `U`. If a `static_cast<U>(t)` is
 * possible, use that, otherwise use `dynamic_cast<U>(t)`.
 */
template <typename U, typename T>
U fast_pointer_cast(T* t) {
  if constexpr (detail::can_static_cast_v<T*, U>) {
    return static_cast<U>(t);
  } else {
    return dynamic_cast<U>(t);
  }
}
}  // namespace tt
