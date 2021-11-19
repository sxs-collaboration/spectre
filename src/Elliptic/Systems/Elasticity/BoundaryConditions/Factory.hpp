// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Elliptic/BoundaryConditions/AnalyticSolution.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Elliptic/Systems/Elasticity/BoundaryConditions/LaserBeam.hpp"
#include "Elliptic/Systems/Elasticity/BoundaryConditions/Zero.hpp"
#include "Utilities/TMPL.hpp"

namespace Elasticity::BoundaryConditions {

template <typename System>
using standard_boundary_conditions = tmpl::append<
    tmpl::list<
        elliptic::BoundaryConditions::AnalyticSolution<System>,
        Zero<System::volume_dim, elliptic::BoundaryConditionType::Dirichlet>,
        Zero<System::volume_dim, elliptic::BoundaryConditionType::Neumann>>,
    tmpl::conditional_t<System::volume_dim == 3, tmpl::list<LaserBeam>,
                        tmpl::list<>>>;

}
