// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryConditions/DemandOutgoingCharSpeeds.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryConditions/Reflection.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianMhd::BoundaryConditions {
/// Typelist of standard BoundaryConditions
template <size_t Dim>
using standard_boundary_conditions =
    tmpl::list<domain::BoundaryConditions::Periodic<BoundaryCondition<Dim>>,
               DemandOutgoingCharSpeeds<Dim>, DirichletAnalytic<Dim>,
               Reflection<Dim>>;
}  // namespace NewtonianMhd::BoundaryConditions
