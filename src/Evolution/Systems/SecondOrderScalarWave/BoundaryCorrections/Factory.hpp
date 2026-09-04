// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Evolution/Systems/SecondOrderScalarWave/BoundaryCorrections/LaxFriedrichs.hpp"
#include "Utilities/TMPL.hpp"

namespace SecondOrderScalarWave::BoundaryCorrections {
template <size_t Dim>
using standard_boundary_corrections = tmpl::list<LaxFriedrichs<Dim>>;
}  // namespace SecondOrderScalarWave::BoundaryCorrections
