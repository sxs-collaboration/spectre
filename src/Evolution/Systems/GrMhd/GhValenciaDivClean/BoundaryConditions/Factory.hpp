// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/ConstraintPreservingFreeOutflow.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/DirichletFreeOutflow.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/Factory.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::GhValenciaDivClean::BoundaryConditions {
/// Boundary conditions that work with finite difference.
using standard_boundary_conditions = tmpl::list<
    ConstraintPreservingFreeOutflow,
    domain::BoundaryConditions::Periodic<BoundaryCondition>,
    grmhd::GhValenciaDivClean::BoundaryConditions::DirichletAnalytic,
    grmhd::GhValenciaDivClean::BoundaryConditions::DirichletFreeOutflow>;
}  // namespace grmhd::GhValenciaDivClean::BoundaryConditions
