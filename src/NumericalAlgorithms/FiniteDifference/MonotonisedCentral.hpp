// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <utility>

#include "NumericalAlgorithms/FiniteDifference/Reconstruct.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim>
class Direction;
template <size_t Dim>
class Index;
/// \endcond

namespace fd::reconstruction {
namespace detail {
struct MonotonisedCentralReconstructor {
  SPECTRE_ALWAYS_INLINE static std::array<double, 2> pointwise(
      const double* const q, const int stride) {
    using std::abs;

    const double a = q[stride] - q[0];
    const double b = q[0] - q[-stride];

    if (sgn(a) != sgn(b)) {
      return {{q[0], q[0]}};
    }
    if (3.0 * abs(a) <= abs(b)) {
      return {{q[0] - a, q[stride]}};
    } else if (3.0 * abs(b) <= abs(a)) {
      return {{q[-stride], q[0] + b}};
    } else {
      const double slope = 0.5 * (q[stride] - q[-stride]);
      return {{q[0] - 0.5 * slope, q[0] + 0.5 * slope}};
    }
  }

  SPECTRE_ALWAYS_INLINE static constexpr size_t stencil_width() { return 3; }
};
}  // namespace detail

/*!
 * \ingroup FiniteDifferenceGroup
 * \brief Performs monotonised central-difference reconstruction on the `vars`
 * in each direction.
 *
 * On a 1d mesh with spacing \f$\Delta x\f$ we denote the solution at the
 * \f$j\f$th point by \f$u_j\f$. The monotonised central-difference
 * reconstruction \cite RezzollaBook first computes:
 *
 * \f{align}
 * \sigma_j\equiv \textrm{minmod}
 *           \left(\frac{u_{j+1} - u_{j-1}}{2\Delta x},
 *                 2\frac{u_j-u_{j-1}}{\Delta x},
 *                 2\frac{u_{j+1}-u_j}{\Delta x}\right)
 * \f}
 *
 * where
 *
 * \f{align}
 *  \mathrm{minmod}(a,b,c)=
 * \left\{
 *   \begin{array}{ll}
 *    \mathrm{sgn}(a)\min(\lvert a\rvert, \lvert b\rvert, \lvert c\rvert)
 *      & \mathrm{if} \; \mathrm{sgn}(a)=\mathrm{sgn}(b)=\mathrm{sgn}(c) \\
 *    0 & \mathrm{otherwise}
 *   \end{array}\right.
 * \f}
 *
 * The reconstructed solution \f$u_j(x)\f$ in the \f$j\f$th cell is given by
 *
 * \f{align}
 * u_j(x)=u_j + \sigma_j (x-x_j)
 * \f}
 *
 * where \f$x_j\f$ is the coordinate \f$x\f$ at the center of the cell.
 */
template <size_t Dim>
void monotonised_central(
    const gsl::not_null<std::array<gsl::span<double>, Dim>*>
        reconstructed_upper_side_of_face_vars,
    const gsl::not_null<std::array<gsl::span<double>, Dim>*>
        reconstructed_lower_side_of_face_vars,
    const gsl::span<const double>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Index<Dim>& volume_extents, const size_t number_of_variables) {
  detail::reconstruct<detail::MonotonisedCentralReconstructor>(
      reconstructed_upper_side_of_face_vars,
      reconstructed_lower_side_of_face_vars, volume_vars, ghost_cell_vars,
      volume_extents, number_of_variables);
}
}  // namespace fd::reconstruction
