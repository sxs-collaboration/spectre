// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>

#include "NumericalAlgorithms/FiniteDifference/Reconstruct.hpp"
#include "NumericalAlgorithms/FiniteDifference/Unlimited.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Math.hpp"

/// \cond
class DataVector;
template <size_t>
class Direction;
template <size_t Dim, typename T>
class DirectionMap;
template <size_t Dim>
class Index;
/// \endcond

namespace fd::reconstruction {
namespace detail {
struct MonotonicityPreserving5Reconstructor {
  SPECTRE_ALWAYS_INLINE static std::array<double, 2> pointwise(
      const double* const q, const int stride, const double alpha,
      const double epsilon) {
    using std::abs;
    using std::max;
    using std::min;

    // define minmod function for 2 and 4 args
    const auto minmod2 = [](const double x, const double y) {
      return 0.5 * (sgn(x) + sgn(y)) * min(abs(x), abs(y));
    };
    const auto minmod4 = [](const double w, const double x, const double y,
                            const double z) {
      const double sign_w = sgn(w);
      return 0.125 * (sign_w + sgn(x)) *
             abs((sign_w + sgn(y)) * (sign_w + sgn(z))) *
             min(abs(w), min(abs(x), min(abs(y), abs(z))));
    };

    // first, compute unlimited fifth-order finite difference reconstruction
    // values
    auto result = UnlimitedReconstructor<4>::pointwise(q, stride);

    // compute q_{j+1/2}
    const double q_mp_plus =
        q[0] + minmod2(q[stride] - q[0], alpha * (q[0] - q[-stride]));
    // compute q_{j-1/2}
    const double q_mp_minus =
        q[0] + minmod2(q[-stride] - q[0], alpha * (q[0] - q[stride]));

    const bool limit_q_plus =
        ((result[1] - q[0]) * (result[1] - q_mp_plus) > epsilon);
    const bool limit_q_minus =
        ((result[0] - q[0]) * (result[0] - q_mp_minus) > epsilon);

    if (limit_q_plus or limit_q_minus) {
      const double dp = q[2 * stride] + q[0] - 2.0 * q[stride];
      const double dj = q[stride] + q[-stride] - 2.0 * q[0];
      const double dm = q[0] + q[-2 * stride] - 2.0 * q[-stride];
      const double dm4_plus = minmod4(4.0 * dj - dp, 4 * dp - dj, dj, dp);
      const double dm4_minus = minmod4(4.0 * dj - dm, 4 * dm - dj, dj, dm);

      if (limit_q_plus) {
        const double q_ul = q[0] + alpha * (q[0] - q[-stride]);
        const double q_md =
            0.5 * (q[0] + q[stride] - dm4_plus);  // inline q^{AV}
        const double q_lc =
            q[0] + 0.5 * (q[0] - q[-stride]) + 1.3333333333333333 * dm4_minus;
        const double q_min =
            max(min(q[0], min(q[stride], q_md)), min(q[0], min(q_ul, q_lc)));
        const double q_max =
            min(max(q[0], max(q[stride], q_md)), max(q[0], max(q_ul, q_lc)));

        result[1] = result[1] + minmod2(q_min - result[1], q_max - result[1]);
      }

      if (limit_q_minus) {
        const double q_ul = q[0] + alpha * (q[0] - q[stride]);
        const double q_md =
            0.5 * (q[0] + q[-stride] - dm4_minus);  // inline q^{AV}
        const double q_lc =
            q[0] + 0.5 * (q[0] - q[stride]) + 1.3333333333333333 * dm4_plus;
        const double q_min =
            max(min(q[0], min(q[-stride], q_md)), min(q[0], min(q_ul, q_lc)));
        const double q_max =
            min(max(q[0], max(q[-stride], q_md)), max(q[0], max(q_ul, q_lc)));

        result[0] = result[0] + minmod2(q_min - result[0], q_max - result[0]);
      }
    }

    return result;
  }

  SPECTRE_ALWAYS_INLINE static constexpr size_t stencil_width() { return 5; }
};
}  // namespace detail

/*!
 * \ingroup FiniteDifferenceGroup
 * \brief Performs the fifth order monotonicity-preserving (MP5) reconstruction
 * \cite Suresh1997.
 *
 * First, calculate the original interface value \f$q_{j+1/2}^\text{OR}\f$ with
 * the (unlimited) fifth order finite difference reconstruction
 *
 * \f{align}
 *  q_{j+1/2}^\text{OR} = \frac{1}{128}( 3 q_{j-2} - 20 q_{j-1} + 90 q_{j}
 *            + 60 q_{j+1} - 5 q_{j+2}) .
 * \f}
 *
 * and compute
 *
 * \f{align}
 *  q^\text{MP} = q_j + \text{minmod}(q_{j+1} - q_j, \alpha(q_j - q_{j-1}))
 * \f}
 *
 * for a given value of \f$\alpha\f$, which is usually set to \f$\alpha=4\f$.
 *
 * If \f$ (q_{j+1/2}^\text{OR} - q_j)(q_{j+1/2}^\text{OR} - q^\text{MP}) \leq
 * \epsilon\f$ where \f$\epsilon\f$ is a small tolerance value, use $q_{1/2}
 * = q_{j+1/2}^\text{OR}$.
 *
 * \note A proper value of \f$\epsilon\f$ may depend on the scale of the
 * quantity \f$q\f$ to be reconstructed; reference \cite Suresh1997 suggests
 * 1e-10. For hydro simulations with atmosphere treatment, setting
 * \f$\epsilon=0.0\f$ would be safe.
 *
 * Otherwise, calculate
 *
 * \f{align}
 *  d_{j+1} & = q_{j+2} - 2q_{j+1} + q_{j}       \\
 *  d_j     & = q_{j+1} - 2q_j + q_{j-1}         \\
 *  d_{j-1} & = q_{j} - 2q_{j-1} + q_{j-2} ,
 * \f}
 *
 * \f{align}
 *  d^\text{M4}_{j+1/2} =
 *                \text{minmod}(4d_j - d_{j+1}, 4d_{j+1}-d_j, d_j, d_{j+1}) \\
 *  d^\text{M4}_{j-1/2} =
 *                \text{minmod}(4d_j - d_{j-1}, 4d_{j-1}-d_j, d_j, d_{j-1}),
 * \f}
 *
 * \f{align}
 *  q^\text{UL} & = q_j + \alpha(q_j - q_{j-1})                             \\
 *  q^\text{AV} & = (q_j + q_{j+1}) / 2                                     \\
 *  q^\text{MD} & = q^\text{AV} -  d^\text{M4}_{j+1/2} / 2                  \\
 *  q^\text{LC} & = q_j + (q_j - q_{j-1}) / 2 + (4/3) d^\text{M4}_{j-1/2}
 * \f}
 *
 * and
 * \f{align}
 *  q^\text{min} & = \text{max}[ \text{min}(q_j, q_{j+1}, q^\text{MD}),
 *                               \text{min}(q_j, q^\text{UL}, q^\text{LC}) ] \\
 *  q^\text{max} & = \text{min}[ \text{max}(q_j, q_{j+1}, q^\text{MD}),
 *                               \text{max}(q_j, q^\text{UL}, q^\text{LC}) ].
 * \f}
 *
 * The reconstructed value is given as
 * \f{align}
 *  q_{i+1/2} = \text{median}(q_{j+1/2}^\text{OR},  q^\text{min}, q^\text{max})
 * \f}
 * where the median function can be expressed as
 * \f{align}
 *  \text{median}(x,y,z) = x + \text{minmod}(y-x, z-x).
 * \f}
 *
 */
template <size_t Dim>
void monotonicity_preserving_5(
    const gsl::not_null<std::array<gsl::span<double>, Dim>*>
        reconstructed_upper_side_of_face_vars,
    const gsl::not_null<std::array<gsl::span<double>, Dim>*>
        reconstructed_lower_side_of_face_vars,
    const gsl::span<const double>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Index<Dim>& volume_extents, const size_t number_of_variables,
    const double alpha, const double epsilon) {
  detail::reconstruct<detail::MonotonicityPreserving5Reconstructor>(
      reconstructed_upper_side_of_face_vars,
      reconstructed_lower_side_of_face_vars, volume_vars, ghost_cell_vars,
      volume_extents, number_of_variables, alpha, epsilon);
}
}  // namespace fd::reconstruction
