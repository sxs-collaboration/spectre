// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>

#include "Utilities/ForceInline.hpp"
#include "Utilities/Requires.hpp"

/*!
 * \ingroup UtilitiesGroup
 * \brief Returns the `i`th element if `T` has a subscript operator, otherwise
 * if `T` is fundamental returns `t`.
 */
template <class T>
SPECTRE_ALWAYS_INLINE auto get_element(T& t, const size_t i) noexcept
    -> decltype(t[i]) {
  return t[i];
}

/// \cond
template <class T, Requires<std::is_fundamental<T>::value> = nullptr>
SPECTRE_ALWAYS_INLINE T& get_element(T& t, size_t /*i*/) noexcept {
  return t;
}
/// \endcond

/*!
 * \ingroup UtilitiesGroup
 * \brief Retrieve the size of `t` if `t.size()` is a valid expression,
 * otherwise if `T` is fundamental returns 1
 */
template <class T>
SPECTRE_ALWAYS_INLINE auto get_size(const T& t) noexcept -> decltype(t.size()) {
  return t.size();
}

/// \cond
template <class T, Requires<std::is_fundamental<T>::value> = nullptr>
SPECTRE_ALWAYS_INLINE size_t get_size(const T& /*t*/) noexcept {
  return 1;
}
/// \endcond
