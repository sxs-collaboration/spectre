// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <limits>

#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ForceInline.hpp"

/// \ingroup UtilitiesGroup
/// Checks if two values `a` and `b` are equal within roundoff,
/// by comparing `abs(a - b) < (max(abs(a), abs(b)) + scale) * eps`.
constexpr SPECTRE_ALWAYS_INLINE bool equal_within_roundoff(
    const double a, const double b,
    const double eps = std::numeric_limits<double>::epsilon() * 100.0,
    const double scale = 1.0) noexcept {
  return ce_fabs(a - b) < (std::max(ce_fabs(a), ce_fabs(b)) + scale) * eps;
}
