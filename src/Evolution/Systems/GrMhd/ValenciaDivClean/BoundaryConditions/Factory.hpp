// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/Outflow.hpp"

namespace grmhd::ValenciaDivClean::BoundaryConditions {
/// Typelist of standard BoundaryConditions
using standard_boundary_conditions =
    tmpl::list<DirichletAnalytic, Outflow,
               domain::BoundaryConditions::Periodic<BoundaryCondition>>;
}  // namespace grmhd::ValenciaDivClean::BoundaryConditions
