// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Evolution/Systems/CurvedScalarWave/BoundaryCorrections/UpwindPenalty.hpp"
#include "Utilities/TMPL.hpp"

namespace CurvedScalarWave::BoundaryCorrections {
template <size_t Dim>
using standard_boundary_corrections = tmpl::list<UpwindPenalty<Dim>>;
}  // namespace CurvedScalarWave::BoundaryCorrections
