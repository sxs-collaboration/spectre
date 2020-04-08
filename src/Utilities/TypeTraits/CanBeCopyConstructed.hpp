// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Utilities/TypeTraits.hpp"

namespace tt {
// @{
/*!
 * \ingroup TypeTraitsGroup
 * \brief Check if `T` is copy constructible
 *
 * The STL `std::is_copy_constructible` does not work as expected with some
 * types, such as `std::unordered_map`. This is because
 * `std::is_copy_constructible` only checks that the copy construction call is
 * well-formed, not that it could actually be done in practice. To get around
 * this for containers we check that `T::value_type` is also copy constructible.
 */
template <typename T, typename = void>
struct can_be_copy_constructed : std::is_copy_constructible<T> {};

/// \cond
template <typename T>
struct can_be_copy_constructed<T, std::void_t<typename T::value_type>>
    : std::bool_constant<std::is_copy_constructible_v<T> and
                         std::is_copy_constructible_v<typename T::value_type>> {
};
/// \endcond

template <typename T>
constexpr bool can_be_copy_constructed_v = can_be_copy_constructed<T>::value;
// @}
}  // namespace tt
