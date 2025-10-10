// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hll.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Rusanov.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::BoundaryCorrections {
using standard_boundary_corrections = tmpl::list<Hll, Rusanov>;
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
