// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class Poisson::FirstOrderSystem

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Elliptic/Systems/Poisson/Equations.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"
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

namespace Poisson {

/*!
 * \brief The Poisson equation formulated as a set of coupled first-order PDEs.
 *
 * \details This system formulates the Poisson equation \f$-\Delta u(x) =
 * f(x)\f$ as the set of coupled first-order PDEs
 * \f[
 * -\nabla \cdot \boldsymbol{v}(x) = f(x) \\
 * \nabla u(x) - \boldsymbol{v}(x) = 0
 * \f]
 * where we make use of an auxiliary variable \f$\boldsymbol{v}\f$. This scheme
 * also goes by the name of _mixed_ or _flux_ formulation (see e.g.
 * \cite Arnold2002). The auxiliary variable is treated on the same footing as
 * the field \f$u\f$. This allows us to make use of the DG architecture
 * developed for coupled first-order PDEs, in particular the flux communication
 * and lifting code. It does, however, introduce auxiliary degrees of freedom
 * that can be avoided in the _primal formulation_. Furthermore, the linear
 * operator that represents the DG discretization for this system is not
 * symmetric (since no mass operator is applied) and has both positive and
 * negative eigenvalues. These properties further increase the computational
 * cost (see \ref LinearSolverGroup) and are remedied in the primal formulation.
 */
template <size_t Dim>
struct FirstOrderSystem {
  static constexpr size_t volume_dim = Dim;

  // The physical fields to solve for
  using fields_tag = Tags::Variables<tmpl::list<Field, AuxiliaryField<Dim>>>;
  using impose_boundary_conditions_on_fields = tmpl::list<Field>;

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
  using magnitude_tag = Tags::EuclideanMagnitude<Tag>;

  // The tags to instantiate derivative functions for
  using gradient_tags = tmpl::list<LinearSolver::Tags::Operand<Field>>;
  using divergence_tags = tmpl::list<AuxiliaryField<Dim>>;
};
}  // namespace Poisson
