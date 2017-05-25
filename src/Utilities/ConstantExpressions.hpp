// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Define simple functions for constant expressions.

#pragma once

#include <type_traits>

#include "Utilities/ForceInline.hpp"

/// \ingroup ConstantExpressions
/// Compute 2 to the n for integral types.
///
/// \param n the power of two to compute.
/// \return 2^n
template <typename T,
          typename = typename std::enable_if<std::is_integral<T>::value>::type>
SPECTRE_ALWAYS_INLINE constexpr T two_to_the(T n) {
  return static_cast<T>(1) << n;
}
