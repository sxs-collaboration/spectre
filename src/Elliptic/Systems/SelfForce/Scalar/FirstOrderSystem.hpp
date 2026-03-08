// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/Protocols/FirstOrderSystem.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/Equations.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/Tags.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarSelfForce {

/*!
 * \brief Self-force of a scalar charge on a Kerr background.
 *
 * In this formulation we solve a 2D elliptic equation for the m-mode field
 * $\Psi_m$, following \cite Osburn:2022bby . The two dimensions are a radial
 * and angular coordinate, specifically the tortoise radius $r_\star$ and
 * $\cos\theta$ (note that \cite Osburn:2022bby uses $\theta$).
 *
 * We write the equations in generic first-order flux form
 * \begin{equation}
 * -\Delta_m \Psi_m = -\partial_i F^i + \beta \Psi_m + \gamma_i \partial_i
 * \Psi_m = 0
 * \end{equation}
 * with the flux
 * \begin{equation}
 * F^i = \{\partial_{r_\star}, \alpha \partial_{\cos\theta}\} \Psi_m
 * \text{,}
 * \end{equation}
 * where $\alpha$, $\beta$, and $\gamma_i$ are coefficients that define the
 * elliptic equations. The particular coefficients that match Eq. (2.9) of
 * \cite Osburn:2022bby for a circular equatorial orbit in Kerr are implemented
 * in `ScalarSelfForce::AnalyticData::CircularOrbit`.
 *
 * \par Regularization and modified boundary data
 * As described in \cite Osburn:2022bby Sec. III, the field $\Psi_m$ is singular
 * at the location of the scalar point charge ("puncture"). Therefore, in a
 * region near the puncture we split the field into a regular and a singular
 * part,
 * \begin{equation}
 *   \Psi_m = \Psi_m^R + \Psi_m^P
 *   \text{,}
 * \end{equation}
 * where $\Psi_m^P$ is the singular part that we can (approximately) compute
 * analytically and $\Psi_m^R$ is the regular part that we solve for. Therefore,
 * we need to transform variables between the full field $\Psi_m$ and the
 * regularized field $\Psi_m^R$ when data moves across the boundary of the
 * regularized region. We do this at element boundaries using the
 * `modify_boundary_data` mechanism detailed in
 * `elliptic::protocols::FirstOrderSystem` by just adding or subtracting the
 * singular field to/from the received data (see also
 * `ScalarSelfForce::ModifyBoundaryData`). In this regularized region the
 * equations transform to
 * \begin{equation}
 * -\Delta_m \Psi_m^R = \Delta_m \Psi_m^P = S_m^\mathrm{eff}
 * \text{,}
 * \end{equation}
 * so they gain an effective source $S_m^\mathrm{eff}$ from the singular field
 * $\Psi_m^P$. The singular field and the effective source for a circular
 * equatorial orbit in Kerr is implemented in
 * `ScalarSelfForce::AnalyticData::CircularOrbit`.
 */
struct FirstOrderSystem
    : tt::ConformsTo<elliptic::protocols::FirstOrderSystem> {
  static constexpr size_t volume_dim = 2;

  using primal_fields = tmpl::list<Tags::MMode>;
  using primal_fluxes =
      tmpl::list<::Tags::Flux<Tags::MMode, tmpl::size_t<2>, Frame::Inertial>>;

  using background_fields = tmpl::list<Tags::Alpha, Tags::Beta, Tags::Gamma>;
  using inv_metric_tag = void;

  using fluxes_computer = Fluxes;
  using sources_computer = Sources;
  using modify_boundary_data = ModifyBoundaryData;

  using boundary_conditions_base =
      elliptic::BoundaryConditions::BoundaryCondition<2>;
};

}  // namespace ScalarSelfForce
