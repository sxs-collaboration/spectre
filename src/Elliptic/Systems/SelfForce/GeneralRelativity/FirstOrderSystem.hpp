// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/Protocols/FirstOrderSystem.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/Equations.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/Tags.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace GrSelfForce {

/*!
 * \brief Gravitational self-force of a small gravitating body in a Kerr
 * background.
 *
 * Extension of the `ScalarSelfForce::FirstOrderSystem` to the gravitational
 * case. We solve a 2D elliptic equation for the 10 independent components of
 * the symmetric m-mode field $(\Psi_m)_{ab}$. The two dimensions are a
 * radial and angular coordinate, specifically the tortoise radius $r_\star$ and
 * polar angle $\theta$. We parametrize the 2D elliptic equations as:
 * \begin{equation}
 * -\Delta_m (\Psi_m)_{ab} = -\partial_i F^i_{ab} + \beta_{ab}^{cd}
 *   (\Psi_m)_{cd} + \gamma_{iab}^{cd} F^i_{cd} = 0
 * \end{equation}
 * with the flux
 * \begin{equation}
 * F^i_{ab} = \{\partial_{r_\star}, \alpha \partial_\theta\} (\Psi_m)_{ab}
 * \text{,}
 * \end{equation}
 * where $\alpha$, $\beta_{ab}^{cd}$, and $\gamma_{iab}^{cd}$ are coefficients
 * that define the elliptic equations. The particular coefficients for a
 * circular equatorial orbit in Kerr are implemented in
 * `GrSelfForce::AnalyticData::CircularOrbit`.
 *
 * See `ScalarSelfForce::FirstOrderSystem` for a description of the
 * regularization and modified boundary data.
 *
 * Once the 2D elliptic equations are solved for a given m-mode, the
 * contribution to the self-force can be extracted from the gradient of
 * $(\Psi_m)_{ab}$ at the location of the small body. These self-force
 * contributions are then summed over all m-modes up to some cutoff. The
 * resulting self-force can be used to drive a quasi-adiabatic inspiral to
 * generate waveforms. For details and more references see \cite Osburn:2022bby
 * for now (more references will be added when they are published).
 */
struct FirstOrderSystem
    : tt::ConformsTo<elliptic::protocols::FirstOrderSystem> {
  static constexpr size_t volume_dim = 2;

  using primal_fields = tmpl::list<Tags::MMode>;
  using primal_fluxes =
      tmpl::list<::Tags::Flux<Tags::MMode, tmpl::size_t<2>, Frame::Inertial>>;

  using background_fields =
      tmpl::list<Tags::Alpha, Tags::Beta, Tags::GammaRstar, Tags::GammaTheta>;
  using inv_metric_tag = void;

  using fluxes_computer = Fluxes;
  using sources_computer = Sources;
  using modify_boundary_data = ModifyBoundaryData;

  using boundary_conditions_base =
      elliptic::BoundaryConditions::BoundaryCondition<2>;
};

}  // namespace GrSelfForce
