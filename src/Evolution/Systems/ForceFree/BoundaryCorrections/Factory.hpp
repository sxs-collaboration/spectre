// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/ForceFree/BoundaryCorrections/Rusanov.hpp"
#include "Utilities/TMPL.hpp"

namespace ForceFree::BoundaryCorrections {
using standard_boundary_corrections = tmpl::list<Rusanov>;
}  // namespace ForceFree::BoundaryCorrections
