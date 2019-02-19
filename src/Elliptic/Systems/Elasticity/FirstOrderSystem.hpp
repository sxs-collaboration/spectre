// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Elliptic/Systems/Elasticity/Equations.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
template <typename>
class Variables;
}  // namespace Tags

namespace LinearSolver {
namespace Tags {
template <typename>
struct Operand;
}  // namespace Tags
}  // namespace LinearSolver
/// \endcond

namespace Elasticity {

/*!
 * \brief The linear elasticity equations formulated as a set of coupled
 * first-order PDEs.
 *
 * This system formulates the linear elasticity problem
 * \f$-\nabla_i Y^{ijkl}\nabla_{(k}u_{l)}=f_\mathrm{ext}^j\f$ (see `Elasticity`)
 * as the set of coupled first-order PDEs
 * \f[
 * \nabla_i T^{ij} = f_\mathrm{ext}^j \\
 * -Y^{ijkl}\nabla_{(k}u_{l)} - T^{ij} = 0
 * \f]
 * by choosing the stress \f$T^{ij}=-Y^{ijkl}\nabla_{(k}u_{l)}\f$ as an
 * auxiliary variable. See the `Poisson::FirstOrderSystem` for a discussion of
 * alternatives to the first-order formulation.
 */
template <size_t Dim>
struct FirstOrderSystem {
  static constexpr size_t volume_dim = Dim;

  // The physical fields to solve for
  using fields_tag =
      ::Tags::Variables<tmpl::list<Tags::Displacement<Dim>, Tags::Stress<Dim>>>;
  using impose_boundary_conditions_on_fields =
      tmpl::list<Tags::Displacement<Dim>>;

  // The variables to compute bulk contributions and fluxes for.
  using variables_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;

  // The bulk contribution to the linear operator action
  using compute_operator_action = ComputeFirstOrderOperatorAction<Dim>;

  // The interface normal dotted into the fluxes that is required by the strong
  // flux lifting operation
  using normal_dot_fluxes = ComputeFirstOrderNormalDotFluxes<Dim>;

  // The tag of the operator to compute magnitudes on the manifold, e.g. to
  // normalize vectors on the faces of an element
  template <typename Tag>
  using magnitude_tag = ::Tags::EuclideanMagnitude<Tag>;

  // The tags to instantiate derivative functions for
  using gradient_tags =
      tmpl::list<LinearSolver::Tags::Operand<Tags::Displacement<Dim>>>;
  using divergence_tags = tmpl::list<Tags::Stress<Dim>>;
};

}  // namespace Elasticity
