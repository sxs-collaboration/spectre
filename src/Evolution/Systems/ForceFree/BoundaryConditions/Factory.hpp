// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Evolution/Systems/ForceFree/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/ForceFree/BoundaryConditions/DirichletAnalytic.hpp"
#include "Utilities/TMPL.hpp"

namespace ForceFree::BoundaryConditions {
/// Typelist of standard BoundaryConditions
using standard_boundary_conditions =
    tmpl::list<domain::BoundaryConditions::Periodic<BoundaryCondition>,
               DirichletAnalytic>;
}  // namespace ForceFree::BoundaryConditions
