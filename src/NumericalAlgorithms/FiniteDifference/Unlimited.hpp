// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "NumericalAlgorithms/FiniteDifference/Reconstruct.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
template <size_t Dim, typename T>
class DirectionMap;
template <size_t Dim>
class Index;
/// \endcond

namespace fd::reconstruction {
namespace detail {
template <size_t Degree>
struct UnlimitedReconstructor {
  static_assert(Degree == 2 or Degree == 4);
  SPECTRE_ALWAYS_INLINE static std::array<double, 2> pointwise(
      const double* const q, const int stride) {
    if constexpr (Degree == 2) {
      // quadratic polynomial
      return {{0.375 * q[-stride] + 0.75 * q[0] - 0.125 * q[stride],
               -0.125 * q[-stride] + 0.75 * q[0] + 0.375 * q[stride]}};
    } else if constexpr (Degree == 4) {
      // quintic polynomial
      return {{-0.0390625 * q[-2 * stride] + 0.46875 * q[-stride] +
               0.703125 * q[0] - 0.15625 * q[stride] +
               0.0234375 * q[2 * stride],
               0.0234375 * q[-2 * stride] - 0.15625 * q[-stride] +
               0.703125 * q[0] + 0.46875 * q[stride] -
               0.0390625 * q[2 * stride]}};
    }
  }

  SPECTRE_ALWAYS_INLINE static constexpr size_t stencil_width() {
    return Degree + 1;
  }
};
}  // namespace detail

/*!
 * \ingroup FiniteDifferenceGroup
 * \brief Performs unlimited reconstruction on the `vars` in each direction.
 *
 * On a 1d mesh with spacing \f$\Delta x\f$ we denote the solution at the
 * \f$j\f$th point by \f$u_j\f$. The degree 2 reconstruction is
 *
 * \f{align}{
 * u_{j+1/2} &= -\frac{1}{8} u_{j-1} + \frac{3}{4} u_j + \frac{3}{8} u_{j+1}, \\
 * u_{j-1/2} &= \frac{3}{8} u_{j-1} + \frac{3}{4} u_j - \frac{1}{8} u_{j+1}.
 * \f}
 *
 * The degree 4 reconstruction is
 *
 * \f{align}{
 *  u_{j-1/2} &= - \frac{5}{128}u_{j-2} + \frac{15}{32}u_{j-1} +
 *            \frac{45}{64}u_{j} -\frac{5}{32}u_{j+1} + \frac{3}{128}u_{j+2}, \\
 *  u_{j+1/2} &= \frac{3}{128}u_{j-2} - \frac{5}{32}u_{j-1} + \frac{45}{64}u_{j}
 *            + \frac{15}{32}u_{j+1} - \frac{5}{128}u_{j+2}.
 * \f}
 */
template <size_t Degree, size_t Dim>
void unlimited(
    const gsl::not_null<std::array<gsl::span<double>, Dim>*>
    reconstructed_upper_side_of_face_vars,
    const gsl::not_null<std::array<gsl::span<double>, Dim>*>
    reconstructed_lower_side_of_face_vars,
    const gsl::span<const double>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Index<Dim>& volume_extents, const size_t number_of_variables) {
  detail::reconstruct<detail::UnlimitedReconstructor<Degree>>(
      reconstructed_upper_side_of_face_vars,
      reconstructed_lower_side_of_face_vars, volume_vars, ghost_cell_vars,
      volume_extents, number_of_variables);
}
}  // namespace fd::reconstruction
