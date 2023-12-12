// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/Protocols/FirstOrderSystem.hpp"
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
 * This system formulates the elasticity equation (see `Elasticity`):
 *
 * \f{align*}
 * \nabla_i T^{ij} = f_\mathrm{ext}^j \\
 * T^{ij} = -Y^{ijkl} \nabla_{(k} \xi_{l)}
 * \f}
 *
 * The system can be formulated in terms of these fluxes and sources (see
 * `elliptic::protocols::FirstOrderSystem`):
 *
 * \f{align*}
 * F^{ij} &= -T^{ij} = Y^{ijkl} \nabla_{(k} \xi_{l)} \\
 * S^j &= 0 \\
 * f^j &= f_\mathrm{ext}^j \text{.}
 * \f}
 */
template <size_t Dim>
struct FirstOrderSystem
    : tt::ConformsTo<elliptic::protocols::FirstOrderSystem> {
  static constexpr size_t volume_dim = Dim;

  using primal_fields = tmpl::list<Tags::Displacement<Dim>>;
  using primal_fluxes = tmpl::list<Tags::MinusStress<Dim>>;

  using background_fields = tmpl::list<>;
  using inv_metric_tag = void;

  using fluxes_computer = Fluxes<Dim>;
  using sources_computer = void;

  using boundary_conditions_base =
      elliptic::BoundaryConditions::BoundaryCondition<Dim>;
};
}  // namespace Elasticity
