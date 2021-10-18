// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <limits>
#include <type_traits>

#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/TypeTraits/IsIterable.hpp"
#include "Utilities/TypeTraits/IsMaplike.hpp"

namespace EqualWithinRoundoffImpls {
/*!
 * \brief Specialize this class to add support for the `equal_within_roundoff`
 * function.
 *
 * Ensure the `Lhs` and `Rhs` are symmetric. A specialization must implement a
 * static `apply` function with this signature:
 *
 * ```cpp
 * static bool apply(const Lhs& lhs, const Rhs& rhs, const double eps,
 *                   const double scale);
 * ```
 *
 * It can be helpful to invoke the `equal_within_roundoff` function for floating
 * points from within your specialization.
 */
template <typename Lhs, typename Rhs, typename = std::nullptr_t>
struct EqualWithinRoundoffImpl;
}  // namespace EqualWithinRoundoffImpls

/*!
 * \ingroup UtilitiesGroup
 * \brief Checks if two values `lhs` and `rhs` are equal within roundoff, by
 * comparing `abs(lhs - rhs) < (max(abs(lhs), abs(rhs)) + scale) * eps`.
 *
 * The two values can be floating-point numbers, or any types for which
 * `EqualWithinRoundoffImpls::EqualWithinRoundoffImpl` has been specialized. For
 * example, a default implementation exists for the case where `lhs`, `rhs`, or
 * both, are iterable, and compares the values point-wise.
 */
template <typename Lhs, typename Rhs>
constexpr SPECTRE_ALWAYS_INLINE bool equal_within_roundoff(
    const Lhs& lhs, const Rhs& rhs,
    const double eps = std::numeric_limits<double>::epsilon() * 100.0,
    const double scale = 1.0) {
  return EqualWithinRoundoffImpls::EqualWithinRoundoffImpl<Lhs, Rhs>::apply(
      lhs, rhs, eps, scale);
}

/// Specializations of `EqualWithinRoundoffImpl` for custom types, to add
/// support for the `equal_within_roundoff` function.
namespace EqualWithinRoundoffImpls {

// Compare two floating points
template <typename Floating>
struct EqualWithinRoundoffImpl<Floating, Floating,
                               Requires<std::is_floating_point_v<Floating>>> {
  static constexpr SPECTRE_ALWAYS_INLINE bool apply(const Floating& lhs,
                                                    const Floating& rhs,
                                                    const double eps,
                                                    const double scale) {
    return ce_fabs(lhs - rhs) <=
           (std::max(ce_fabs(lhs), ce_fabs(rhs)) + scale) * eps;
  }
};

// Compare a complex number to a floating point, interpreting the latter as a
// real number
template <typename Floating>
struct EqualWithinRoundoffImpl<std::complex<Floating>, Floating,
                               Requires<std::is_floating_point_v<Floating>>> {
  static SPECTRE_ALWAYS_INLINE bool apply(const std::complex<Floating>& lhs,
                                          const Floating& rhs, const double eps,
                                          const double scale) {
    return equal_within_roundoff(lhs.real(), rhs, eps, scale) and
           equal_within_roundoff(lhs.imag(), 0., eps, scale);
  }
};

// Compare a floating point to a complex number, interpreting the former as a
// real number
template <typename Floating>
struct EqualWithinRoundoffImpl<Floating, std::complex<Floating>,
                               Requires<std::is_floating_point_v<Floating>>> {
  static SPECTRE_ALWAYS_INLINE bool apply(const std::complex<Floating>& lhs,
                                          const Floating& rhs, const double eps,
                                          const double scale) {
    return equal_within_roundoff(rhs, lhs, eps, scale);
  }
};

// Compare two complex numbers
template <typename Floating>
struct EqualWithinRoundoffImpl<std::complex<Floating>, std::complex<Floating>,
                               Requires<std::is_floating_point_v<Floating>>> {
  static SPECTRE_ALWAYS_INLINE bool apply(const std::complex<Floating>& lhs,
                                          const std::complex<Floating>& rhs,
                                          const double eps,
                                          const double scale) {
    return equal_within_roundoff(lhs.real(), rhs.real(), eps, scale) and
           equal_within_roundoff(lhs.imag(), rhs.imag(), eps, scale);
  }
};

// Compare an iterable to a floating point
template <typename Lhs, typename Rhs>
struct EqualWithinRoundoffImpl<
    Lhs, Rhs,
    Requires<tt::is_iterable_v<Lhs> and not tt::is_maplike_v<Lhs> and
             std::is_floating_point_v<Rhs>>> {
  static SPECTRE_ALWAYS_INLINE bool apply(const Lhs& lhs, const Rhs& rhs,
                                          const double eps,
                                          const double scale) {
    return alg::all_of(lhs, [&rhs, &eps, &scale](const auto& lhs_element) {
      return equal_within_roundoff(lhs_element, rhs, eps, scale);
    });
  }
};

// Compare a floating point to an iterable
template <typename Lhs, typename Rhs>
struct EqualWithinRoundoffImpl<
    Lhs, Rhs,
    Requires<tt::is_iterable_v<Rhs> and not tt::is_maplike_v<Rhs> and
             std::is_floating_point_v<Lhs>>> {
  static SPECTRE_ALWAYS_INLINE bool apply(const Lhs& lhs, const Rhs& rhs,
                                          const double eps,
                                          const double scale) {
    return equal_within_roundoff(rhs, lhs, eps, scale);
  }
};

// Compare two iterables
template <typename Lhs, typename Rhs>
struct EqualWithinRoundoffImpl<
    Lhs, Rhs,
    Requires<tt::is_iterable_v<Lhs> and not tt::is_maplike_v<Lhs> and
             tt::is_iterable_v<Rhs> and not tt::is_maplike_v<Rhs>>> {
  static bool apply(const Lhs& lhs, const Rhs& rhs, const double eps,
                    const double scale) {
    auto lhs_it = lhs.begin();
    auto rhs_it = rhs.begin();
    while (lhs_it != lhs.end() and rhs_it != rhs.end()) {
      if (not equal_within_roundoff(*lhs_it, *rhs_it, eps, scale)) {
        return false;
      }
      ++lhs_it;
      ++rhs_it;
    }
    ASSERT(lhs_it == lhs.end() and rhs_it == rhs.end(),
           "Can't compare lhs and rhs because they have different lengths.");
    return true;
  }
};

}  // namespace EqualWithinRoundoffImpls
