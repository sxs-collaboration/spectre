// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <iosfwd>

namespace evolution::BoundaryConditions {
/*!
 * \brief The type of boundary condition.
 *
 * There are generally two categories or types of boundary conditions. Those
 * that:
 *
 * 1. set up additional cells or elements (called ghost cells or ghost elements)
 *    outside the computational domain and then apply boundary corrections the
 *    same way that is done in the the interior. For example, in a discontinuous
 *    Galerkin scheme these would impose boundary conditions through a numerical
 *    flux boundary correction term.
 *
 * 2. change the time derivatives at the external boundary
 */
enum class Type {
  /// Impose boundary conditions by setting values on ghost cells/elements.
  ///
  /// The ghost cell values can come either from internal data or from an
  /// analytic prescription
  Ghost,
  /// Impose boundary conditions on the time derivative of the evolved
  /// variables.
  ///
  /// These are imposed after all internal and external boundaries using a ghost
  /// cell prescription have been handled.
  TimeDerivative,
  /// Impose ghost boundary conditions on some of the evolved variables and time
  /// derivative boundary conditions on others.
  GhostAndTimeDerivative,
  /// Impose outflow boundary conditions on the boundary.
  ///
  /// Typically the outflow boundary conditions should only check that all
  /// characteristic speeds are out of the domain.
  Outflow
};

std::ostream& operator<<(std::ostream& os,
                         Type boundary_condition_type) noexcept;
}  // namespace evolution::BoundaryConditions
