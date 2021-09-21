// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryConditions/ConstraintPreservingSphericalRadiation.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryConditions/Outflow.hpp"
#include "Utilities/TMPL.hpp"

namespace CurvedScalarWave::BoundaryConditions {
/// Typelist of standard BoundaryConditions
template <size_t Dim>
using standard_boundary_conditions =
    tmpl::list<Outflow<Dim>, ConstraintPreservingSphericalRadiation<Dim>,
               domain::BoundaryConditions::Periodic<BoundaryCondition<Dim>>>;
}  // namespace CurvedScalarWave::BoundaryConditions
