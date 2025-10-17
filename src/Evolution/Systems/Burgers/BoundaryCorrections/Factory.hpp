// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/Burgers/BoundaryCorrections/Hll.hpp"
#include "Evolution/Systems/Burgers/BoundaryCorrections/Rusanov.hpp"
#include "Utilities/TMPL.hpp"

namespace Burgers::BoundaryCorrections {
using standard_boundary_corrections = tmpl::list<Hll, Rusanov>;
}  // namespace Burgers::BoundaryCorrections
