// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Elliptic/BoundaryConditions/AnalyticSolution.hpp"
#include "Elliptic/Systems/IrrotationalBns/BoundaryConditions/Neumann.hpp"
#include "Utilities/TMPL.hpp"

namespace IrrotationalBns::BoundaryConditions {

template <typename System>
using standard_boundary_conditions =
    tmpl::list<elliptic::BoundaryConditions::AnalyticSolution<System>, Neumann>;
}
