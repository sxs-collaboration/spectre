// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Elliptic/BoundaryConditions/AnalyticSolution.hpp"
#include "Elliptic/Systems/BnsInitialData/BoundaryConditions/StarSurface.hpp"
#include "Utilities/TMPL.hpp"

/// \brief Boundary conditions for binary neutron star initial data.
namespace BnsInitialData::BoundaryConditions {

template <typename System>
using standard_boundary_conditions =
    tmpl::list<elliptic::BoundaryConditions::AnalyticSolution<System>,
               StarSurface>;
}  // namespace BnsInitialData::BoundaryConditions
