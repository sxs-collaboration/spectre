// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/DemandOutgoingCharSpeeds.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/HydroFreeOutflow.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/Reflective.hpp"

namespace grmhd::ValenciaDivClean::BoundaryConditions {
/// Typelist of standard BoundaryConditions
using standard_boundary_conditions =
    tmpl::list<DemandOutgoingCharSpeeds, DirichletAnalytic, HydroFreeOutflow,
               Reflective,
               domain::BoundaryConditions::Periodic<BoundaryCondition>>;
}  // namespace grmhd::ValenciaDivClean::BoundaryConditions
