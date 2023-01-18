// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Elliptic/Systems/Punctures/BoundaryConditions/Flatness.hpp"
#include "Utilities/TMPL.hpp"

namespace Punctures::BoundaryConditions {

using standard_boundary_conditions = tmpl::list<Flatness>;

}
