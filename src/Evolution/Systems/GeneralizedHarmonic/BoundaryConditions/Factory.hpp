// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Bjorhus.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/DirichletMinkowski.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Outflow.hpp"
#include "Utilities/TMPL.hpp"

namespace GeneralizedHarmonic::BoundaryConditions {
/// Typelist of standard BoundaryConditions
template <size_t Dim>
using standard_boundary_conditions =
    tmpl::list<ConstraintPreservingBjorhus<Dim>, DirichletAnalytic<Dim>,
               DirichletMinkowski<Dim>, Outflow<Dim>,
               domain::BoundaryConditions::Periodic<BoundaryCondition<Dim>>>;
}  // namespace GeneralizedHarmonic::BoundaryConditions
