// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Elliptic/Systems/SelfForce/Scalar/BoundaryConditions/Angular.hpp"
#include "Utilities/TMPL.hpp"

/// Boundary conditions for the scalar self-force system.
namespace ScalarSelfForce::BoundaryConditions {

/// List of all boundary conditions for the scalar self-force system.
using standard_boundary_conditions = tmpl::list<Angular>;

}  // namespace ScalarSelfForce::BoundaryConditions
