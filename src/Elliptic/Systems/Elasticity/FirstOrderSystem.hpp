// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Elliptic/BoundaryConditions/AnalyticSolution.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Elliptic/Protocols/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Elasticity/BoundaryConditions/LaserBeam.hpp"
#include "Elliptic/Systems/Elasticity/BoundaryConditions/Zero.hpp"
#include "Elliptic/Systems/Elasticity/Equations.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/ConstitutiveRelation.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace Elasticity {

/*!
 * \brief The linear elasticity equation formulated as a set of coupled
 * first-order PDEs.
 *
 * This system formulates the elasticity equation \f$\nabla_i T^{ij} =
 * f_\mathrm{ext}^j\f$ (see `Elasticity`). It introduces the symmetric strain
 * tensor \f$S_{kl}\f$ as an auxiliary variable which satisfies the
 * `Elasticity::ConstitutiveRelations` \f$T^{ij} = -Y^{ijkl} S_{kl}\f$ with the
 * material-specific elasticity tensor \f$Y^{ijkl}\f$. Written as a set of
 * coupled first-order PDEs, we get
 *
 * \f{align*}
 * -\nabla_i Y^{ijkl} S_{kl} = f_\mathrm{ext}^j \\
 * -\nabla_{(k} \xi_{l)} + S_{kl} = 0
 * \f}
 *
 * The system can be formulated in terms of these fluxes and sources (see
 * `elliptic::protocols::FirstOrderSystem`):
 *
 * \f{align*}
 * F^i_{\xi^j} &=  Y^{ijkl}_{(\xi, S)} S_{kl} \\
 * S_{\xi^j} &= 0 \\
 * f_{\xi^j} &= f_\mathrm{ext}^j \\
 * F^i_{S_{kl}} &= \delta^{i}_{(k} \xi_{l)} \\
 * S_{S_{kl}} &= S_{kl} \\
 * f_{S_{kl}} &= 0 \text{.}
 * \f}
 */
template <size_t Dim>
struct FirstOrderSystem
    : tt::ConformsTo<elliptic::protocols::FirstOrderSystem> {
 private:
  using displacement = Tags::Displacement<Dim>;
  using strain = Tags::Strain<Dim>;
  using minus_stress = Tags::MinusStress<Dim>;

 public:
  static constexpr size_t volume_dim = Dim;

  using primal_fields = tmpl::list<displacement>;
  using auxiliary_fields = tmpl::list<strain>;

  using primal_fluxes = tmpl::list<minus_stress>;
  using auxiliary_fluxes =
      tmpl::list<::Tags::Flux<strain, tmpl::size_t<Dim>, Frame::Inertial>>;

  using background_fields = tmpl::list<>;
  using inv_metric_tag = void;

  using fluxes_computer = Fluxes<Dim>;
  using sources_computer = Sources<Dim>;

  using boundary_conditions_base =
      elliptic::BoundaryConditions::BoundaryCondition<
          Dim,
          tmpl::append<
              tmpl::list<elliptic::BoundaryConditions::Registrars::
                             AnalyticSolution<FirstOrderSystem>,
                         BoundaryConditions::Registrars::Zero<
                             Dim, elliptic::BoundaryConditionType::Dirichlet>,
                         BoundaryConditions::Registrars::Zero<
                             Dim, elliptic::BoundaryConditionType::Neumann>>,
              tmpl::conditional_t<
                  Dim == 3,
                  tmpl::list<BoundaryConditions::Registrars::LaserBeam>,
                  tmpl::list<>>>>;
};
}  // namespace Elasticity
