// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>

/// @{
/*!
 * \ingroup UtilitiesGroup
 * \brief Returns `t.has_value()` if `t` is a `std::optional` otherwise returns
 * `true`.
 */
template <typename T>
constexpr bool has_value(const T& /*t*/) {
  return true;
}

template <typename T>
constexpr bool has_value(const std::optional<T>& t) {
  return t.has_value();
}
/// @}

/// @{
/*!
 * \ingroup UtilitiesGroup
 * \brief Returns `t.value()` if `t` is a `std::optional` otherwise returns
 * `t`.
 */
template <typename T>
constexpr T& value(T& t) {
  return t;
}

template <typename T>
constexpr const T& value(const T& t) {
  return t;
}

template <typename T>
constexpr const T& value(const std::optional<T>& t) {
  return t.value();
}

template <typename T>
constexpr T& value(std::optional<T>& t) {
  return t.value();
}
/// @}
