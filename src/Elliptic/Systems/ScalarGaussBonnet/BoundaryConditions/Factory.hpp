// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Elliptic/Systems/Poisson/BoundaryConditions/Robin.hpp"
#include "Elliptic/Systems/ScalarGaussBonnet/BoundaryConditions/DoNothing.hpp"
#include "Utilities/TMPL.hpp"

namespace sgb::BoundaryConditions {

using standard_boundary_conditions =
    tmpl::list<::Poisson::BoundaryConditions::Robin<3>, DoNothing>;

}  // namespace sgb::BoundaryConditions
