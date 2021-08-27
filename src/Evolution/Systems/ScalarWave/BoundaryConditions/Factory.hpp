// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Evolution/Systems/ScalarWave/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/ScalarWave/BoundaryConditions/ConstraintPreservingSphericalRadiation.hpp"
#include "Evolution/Systems/ScalarWave/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/ScalarWave/BoundaryConditions/SphericalRadiation.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarWave::BoundaryConditions {
/// Typelist of standard BoundaryConditions
template <size_t Dim>
using standard_boundary_conditions =
    tmpl::list<ConstraintPreservingSphericalRadiation<Dim>,
               DirichletAnalytic<Dim>,
               domain::BoundaryConditions::Periodic<BoundaryCondition<Dim>>,
               SphericalRadiation<Dim>>;
}  // namespace ScalarWave::BoundaryConditions
