// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryConditions/Outflow.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryConditions/Reflection.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianEuler::BoundaryConditions {
/// Typelist of standard BoundaryConditions
template <size_t Dim>
using standard_boundary_conditions =
    tmpl::list<domain::BoundaryConditions::Periodic<BoundaryCondition<Dim>>,
               DirichletAnalytic<Dim>, Outflow<Dim>, Reflection<Dim>>;
}  // namespace NewtonianEuler::BoundaryConditions
