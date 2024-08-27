// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <iosfwd>

/// \cond
namespace Options {
class Option;
template <typename T>
struct create_from_yaml;
}  // namespace Options
/// \endcond

namespace dg {
/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief The DG formulation to use
 *
 * - The `StrongInertial` formulation is also known as the integrate then
 *   transform formulation. The "Inertial" part of the name refers to the fact
 *   that the integration is done over the physical/inertial coordinates, while
 *   the "strong" part refers to the fact that the boundary correction terms are
 *   zero if the solution is continuous at the interfaces.
 *   See \cite Teukolsky2015ega for an overview.
 * - The `WeakInertial` formulation is also known as the integrate then
 *   transform formulation. The "Inertial" part of the name refers to the fact
 *   that the integration is done over the physical/inertial coordinates, while
 *   the "weak" part refers to the fact that the boundary correction terms are
 *   non-zero even if the solution is continuous at the interfaces.
 *   See \cite Teukolsky2015ega for an overview.
 * - The `StrongLogical` formulation is also known as the transform then
 *   integrate formulation. The "logical" part of the name refers to the fact
 *   that the integration is done over the logical coordinates, while the
 *   "strong" part refers to the fact that the boundary correction terms are
 *   zero if the solution is continuous at the interfaces. This formulation
 *   arises from the `StrongInertial` formulation by moving the Jacobians that
 *   appear when computing the divergence of fluxes into the divergence so the
 *   divergence is computed in logical coordinates:
 *   \begin{equation}
 *     \partial_i F^i = \frac{1}{J} \partial_\hat{\imath} J F^\hat{\imath}
 *   \end{equation}
 *   where $J$ is the Jacobian determinant and $\hat{\imath}$ are the logical
 *   coordinates. This is possible because of the "metric identities"
 *   \begin{equation}
 *    \partial_\hat{\imath} \left(J \frac{\partial \xi^\hat{\imath}}
 *    {\partial x^i}\right) = 0.
 *   \end{equation}
 *   See also `dg::metric_identity_det_jac_times_inv_jac` for details and for
 *   functions that compute the Jacobians so they satisfy the metric identities
 *   numerically (which may or may not be useful or necessary).
 */
enum class Formulation { StrongInertial, WeakInertial, StrongLogical };

std::ostream& operator<<(std::ostream& os, Formulation t);
}  // namespace dg

/// \cond
template <>
struct Options::create_from_yaml<dg::Formulation> {
  template <typename Metavariables>
  static dg::Formulation create(const Options::Option& options) {
    return create<void>(options);
  }
};

template <>
dg::Formulation Options::create_from_yaml<dg::Formulation>::create<void>(
    const Options::Option& options);
/// \endcond
