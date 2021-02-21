// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Elliptic/BoundaryConditions/AnalyticSolution.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/Systems/Elasticity/Equations.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/ConstitutiveRelation.hpp"
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
 * The fluxes and sources in terms of the system variables
 * \f$\xi^j\f$ and \f$S_{kl}\f$ are given by
 *
 * \f{align*}
 * F^i_{\xi^j} &=  Y^{ijkl}_{(\xi, S)} S_{kl} \\
 * S_{\xi^j} &= 0 \\
 * f_{\xi^j} &= f_\mathrm{ext}^j \\
 * F^i_{S_{kl}} &= \delta^{i}_{(k} \xi_{l)} \\
 * S_{S_{kl}} &= S_{kl} \\
 * f_{S_{kl}} &= 0 \text{.}
 * \f}
 *
 * See `Poisson::FirstOrderSystem` for details on the first-order
 * flux-formulation.
 */
template <size_t Dim>
struct FirstOrderSystem {
 private:
  using displacement = Tags::Displacement<Dim>;
  using strain = Tags::Strain<Dim>;

 public:
  static constexpr size_t volume_dim = Dim;

  // The physical fields to solve for
  using primal_fields = tmpl::list<displacement>;
  using auxiliary_fields = tmpl::list<strain>;
  using fields_tag =
      ::Tags::Variables<tmpl::append<primal_fields, auxiliary_fields>>;

  // Tags for the first-order fluxes. We can use the symmetric stress here as an
  // optimization once the DG operator supports fluxes with symmetries.
  using primal_fluxes = tmpl::list<
      ::Tags::Flux<displacement, tmpl::size_t<Dim>, Frame::Inertial>>;
  using auxiliary_fluxes =
      tmpl::list<::Tags::Flux<strain, tmpl::size_t<Dim>, Frame::Inertial>>;

  // The variable-independent background fields in the equations
  using background_fields = tmpl::list<>;
  using inv_metric_tag = void;

  // The system equations formulated as fluxes and sources
  using fluxes_computer = Fluxes<Dim>;
  using sources_computer = Sources<Dim>;

  // The supported boundary conditions. Boundary conditions can be
  // factory-created from this base class.
  using boundary_conditions_base =
      elliptic::BoundaryConditions::BoundaryCondition<
          Dim, tmpl::list<elliptic::BoundaryConditions::Registrars::
                              AnalyticSolution<FirstOrderSystem>>>;

  // The tag of the operator to compute magnitudes on the manifold, e.g. to
  // normalize vectors on the faces of an element
  template <typename Tag>
  using magnitude_tag = ::Tags::EuclideanMagnitude<Tag>;
};
}  // namespace Elasticity
