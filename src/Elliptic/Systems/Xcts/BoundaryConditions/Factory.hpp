// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Elliptic/BoundaryConditions/AnalyticSolution.hpp"
#include "Elliptic/Systems/Xcts/BoundaryConditions/ApparentHorizon.hpp"
#include "Elliptic/Systems/Xcts/BoundaryConditions/Flatness.hpp"
#include "Elliptic/Systems/Xcts/BoundaryConditions/Robin.hpp"
#include "Utilities/TMPL.hpp"

namespace Xcts::BoundaryConditions {
template <typename System>
using standard_boundary_conditions =
    tmpl::list<elliptic::BoundaryConditions::AnalyticSolution<System>,
               Flatness<System::enabled_equations>,
               Robin<System::enabled_equations>,
               ApparentHorizon<System::conformal_geometry>>;
}
