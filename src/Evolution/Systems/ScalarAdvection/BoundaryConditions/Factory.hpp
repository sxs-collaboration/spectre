// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Evolution/Systems/ScalarAdvection/BoundaryConditions/BoundaryCondition.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarAdvection::BoundaryConditions {
/// Typelist of standard BoundaryConditions
template <size_t Dim>
using standard_boundary_conditions =
    tmpl::list<domain::BoundaryConditions::Periodic<BoundaryCondition<Dim>>>;
}  // namespace ScalarAdvection::BoundaryConditions
