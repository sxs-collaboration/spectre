// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/Sommerfeld.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::BoundaryConditions {
/// Typelist of standard BoundaryConditions
using standard_boundary_conditions =
    tmpl::list<DirichletAnalytic,
               domain::BoundaryConditions::Periodic<BoundaryCondition>,
               Sommerfeld>;
}  // namespace Ccz4::BoundaryConditions
