// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Evolution/Systems/Burgers/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/Burgers/BoundaryConditions/DemandOutgoingCharSpeeds.hpp"
#include "Evolution/Systems/Burgers/BoundaryConditions/Dirichlet.hpp"
#include "Evolution/Systems/Burgers/BoundaryConditions/DirichletAnalytic.hpp"
#include "Utilities/TMPL.hpp"

namespace Burgers::BoundaryConditions {
/// Typelist of standard BoundaryConditions
using standard_boundary_conditions =
    tmpl::list<DemandOutgoingCharSpeeds, Dirichlet, DirichletAnalytic,
               domain::BoundaryConditions::Periodic<BoundaryCondition>>;
}  // namespace Burgers::BoundaryConditions
