// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "NumericalAlgorithms/FiniteDifference/Reconstruct.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
template <size_t Dim>
class Direction;
template <size_t Dim>
class Index;
/// \endcond

namespace fd::reconstruction {
namespace detail {
struct PpmReconstructor {
  SPECTRE_ALWAYS_INLINE static std::array<double, 2> pointwise(
      const double* const q, const int stride) {
    // Step 1: Unlimited quadratic (degree-2) interpolation
    // u_L = u_{j-1/2} =  3/8 u_{j-1} + 3/4 u_j - 1/8 u_{j+1}
    // u_R = u_{j+1/2} = -1/8 u_{j-1} + 3/4 u_j + 3/8 u_{j+1}
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    const double u_jm1 = q[-stride];
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    const double u_j = q[0];
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    const double u_jp1 = q[stride];

    double u_L = (0.375 * u_jm1) + (0.75 * u_j) - (0.125 * u_jp1);
    double u_R = -(0.125 * u_jm1) + (0.75 * u_j) + (0.375 * u_jp1);

    // Step 2: Colella & Woodward monotonicity limiter (§1.4, adapted to FD)
    // If cell is a local extremum, set both face values to cell center.
    if ((u_R - u_j) * (u_j - u_L) <= 0.0) {
      return {{u_j, u_j}};
    }

    // Apply C&W limiter to prevent overshoot
    const double delta = u_R - u_L;
    const double u_6 = 6.0 * (u_j - (0.5 * (u_L + u_R)));

    if (delta * (delta - u_6) < 0.0) {
      u_L = (3.0 * u_j) - (2.0 * u_R);
    }
    if (delta * (delta + u_6) < 0.0) {
      u_R = (3.0 * u_j) - (2.0 * u_L);
    }

    return {{u_L, u_R}};
  }

  SPECTRE_ALWAYS_INLINE static constexpr size_t stencil_width() { return 3; }
};
}  // namespace detail

/*!
 * \ingroup FiniteDifferenceGroup
 * \brief Performs piecewise parabolic method (PPM) reconstruction on the
 * `volume_vars` in each direction.
 *
 * On a 1d mesh we denote the solution at the \f$j\f$th point by \f$u_j\f$.
 * The PPM reconstruction \cite Colella1984 uses a quadratic interpolation
 * through three points to obtain interface values, which are then monotonicity
 * limited.
 *
 * **Step 1: Unlimited quadratic interpolation**
 *
 * \f{align}
 * u_{j-1/2} &= \frac{3}{8} u_{j-1} + \frac{3}{4} u_j - \frac{1}{8} u_{j+1}
 * \\
 * u_{j+1/2} &= -\frac{1}{8} u_{j-1} + \frac{3}{4} u_j + \frac{3}{8} u_{j+1}
 * \f}
 *
 * **Step 2: Colella & Woodward monotonicity limiter**
 *
 * If the cell is a local extremum,
 * \f$(u_{j+1/2} - u_j)(u_j - u_{j-1/2}) \le 0\f$,
 * both face values are set to the cell center value \f$u_j\f$.
 *
 * Otherwise, define
 * \f{align}
 * \delta u_j &= u_{j+1/2} - u_{j-1/2} \\
 * u_{6,j} &= 6\!\left(u_j - \frac{u_{j-1/2} + u_{j+1/2}}{2}\right)
 * \f}
 *
 * and apply:
 * - If \f$\delta u_j (\delta u_j - u_{6,j}) < 0\f$:
 *   \f$u_{j-1/2} = 3 u_j - 2 u_{j+1/2}\f$
 * - If \f$\delta u_j (\delta u_j + u_{6,j}) < 0\f$:
 *   \f$u_{j+1/2} = 3 u_j - 2 u_{j-1/2}\f$
 *
 * \note This is a finite-difference variant operating on point values, not
 * the finite-volume PPM of \cite Colella1984 which operates on cell averages.
 */
template <size_t Dim>
void ppm(const gsl::not_null<std::array<gsl::span<double>, Dim>*>
             reconstructed_upper_side_of_face_vars,
         const gsl::not_null<std::array<gsl::span<double>, Dim>*>
             reconstructed_lower_side_of_face_vars,
         const gsl::span<const double>& volume_vars,
         const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
         const Index<Dim>& volume_extents, const size_t number_of_variables) {
  detail::reconstruct<detail::PpmReconstructor>(
      reconstructed_upper_side_of_face_vars,
      reconstructed_lower_side_of_face_vars, volume_vars, ghost_cell_vars,
      volume_extents, number_of_variables);
}
}  // namespace fd::reconstruction
