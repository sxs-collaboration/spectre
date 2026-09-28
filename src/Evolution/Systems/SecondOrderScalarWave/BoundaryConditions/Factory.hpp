// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/BoundaryConditions/DirichletCharacteristics.hpp"
#include "Utilities/TMPL.hpp"

namespace SecondOrderScalarWave::BoundaryConditions {
/// Typelist of standard BoundaryConditions
template <size_t Dim>
using standard_boundary_conditions =
    tmpl::list<DirichletAnalytic<Dim>, DirichletCharacteristics<Dim>,
               domain::BoundaryConditions::Periodic<BoundaryCondition<Dim>>>;
}  // namespace SecondOrderScalarWave::BoundaryConditions
