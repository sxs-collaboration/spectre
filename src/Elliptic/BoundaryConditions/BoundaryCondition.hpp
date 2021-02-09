// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic {
/// Boundary conditions for elliptic systems
namespace BoundaryConditions {

/*!
 * \brief Base class for boundary conditions for elliptic systems
 *
 * Boundary conditions for elliptic systems derive from this abstract base
 * class. This allows boundary conditions to be factory-created from input-file
 * options. Specific systems may implement further abstract base classes that
 * derive from this class and add additional requirements.
 *
 * Each derived class represents one kind of boundary conditions. For example,
 * one derived class might implement homogeneous (zero) Dirichlet or Neumann
 * boundary conditions, another might implement Dirichlet fields procured from
 * an analytic solution, and yet another might set the boundary fields as a
 * function of the dynamic variables on the domain boundary (e.g. Robin-type
 * boundary conditions).
 *
 * Note that almost all boundary conditions are actually nonlinear because even
 * those that depend only linearly on the dynamic fields typically contribute
 * non-zero field values. For example, a standard Dirichlet boundary condition
 * \f$u(x=0)=u_0\f$ is nonlinear for any \f$u_0\neq 0\f$. Boundary conditions
 * for linear systems may have exactly this nonlinearity (a constant non-zero
 * contribution) but must depend at most linearly on the dynamic fields.
 * Boundary conditions for nonlinear systems may have any nonlinearity. Either
 * must implement their linearization as a separate function (see below). For
 * linear systems the nonlinear (constant) contribution is typically added to
 * the fixed-source part of the discretized equations and the linearized
 * boundary conditions are being employed throughout the solve so the
 * discretized operator remains linear. For nonlinear systems we typically solve
 * the linearized equations repeatedly for a correction quantity, so we apply
 * the linearized boundary conditions when solving for the correction quantity
 * and apply the nonlinear boundary conditions when dealing with the nonlinear
 * fields that are being corrected (see e.g.
 * `NonlinearSolver::newton_raphson::NewtonRaphson`).
 *
 * Derived classes are expected to implement the following compile-time
 * interface:
 *
 * - They are option-creatable.
 * - They have type aliases `argument_tags`, `volume_tags`,
 *   `argument_tags_linearized` and `volume_tags_linearized`. Those aliases list
 *   the tags required for computing nonlinear and linearized boundary
 *   conditions, respectively. The tags are always taken to represent quantities
 *   on the _interior_ side of the domain boundary, i.e. whenever normal vectors
 *   are involved they point _out_ of the computational domain. The
 *   `volume_tags` list the subset of the `argument_tags` that are _not_
 *   evaluated on the boundary but taken from the element directly.
 * - They have `apply` and `apply_linearized` member functions that take these
 *   arguments (in this order):
 *
 *   1. The dynamic fields as not-null pointers.
 *   2. The normal-dot-fluxes corresponding to the dynamic fields as not-null
 *      pointers. These have the same types as the dynamic fields.
 *   3. The types held by the argument tags.
 *
 *   For first-order systems that involve auxiliary variables, only the
 *   non-auxiliary ("primal") variables are included in the lists above. For
 *   example, boundary conditions for a first-order Poisson system might have an
 *   `apply` function signature that looks like this:
 *
 *   \snippet Elliptic/BoundaryConditions/Test_BoundaryCondition.cpp example_poisson_fields
 *
 * The fields and normal-dot-fluxes passed to the `apply` and `apply_linearized`
 * functions hold data which the implementations of the functions can use, and
 * also serve as return variables. Modifying the fields means applying
 * Dirichlet-type boundary conditions and modifying the normal-dot-fluxes means
 * applying Neumann-type boundary conditions. Just like the arguments evaluated
 * on the boundary, the normal-dot-fluxes involve normal vectors that point
 * _out_ of the computation domain. Note that linearized boundary conditions, as
 * well as nonlinear boundary conditions for linear systems, may only depend
 * linearly on the field data, since these are the fields the linearization is
 * performed for.
 */
template <size_t Dim, typename Registrars>
class BoundaryCondition : public domain::BoundaryConditions::BoundaryCondition {
 private:
  using Base = domain::BoundaryConditions::BoundaryCondition;

 public:
  static constexpr size_t volume_dim = Dim;
  using registrars = Registrars;

  BoundaryCondition() = default;
  BoundaryCondition(const BoundaryCondition&) = default;
  BoundaryCondition(BoundaryCondition&&) = default;
  BoundaryCondition& operator=(const BoundaryCondition&) = default;
  BoundaryCondition& operator=(BoundaryCondition&&) = default;
  ~BoundaryCondition() override = default;

  /// \cond
  explicit BoundaryCondition(CkMigrateMessage* m) noexcept : Base(m) {}
  WRAPPED_PUPable_abstract(BoundaryCondition);  // NOLINT
  /// \endcond

  using creatable_classes = Registration::registrants<registrars>;
};

}  // namespace BoundaryConditions
}  // namespace elliptic
