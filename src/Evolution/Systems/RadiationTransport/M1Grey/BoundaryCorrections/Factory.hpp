// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/RadiationTransport/M1Grey/BoundaryCorrections/Rusanov.hpp"
#include "Utilities/TMPL.hpp"

namespace RadiationTransport::M1Grey::BoundaryCorrections {
template <typename NeutrinoSpeciesList>
using standard_boundary_corrections = tmpl::list<Rusanov<NeutrinoSpeciesList>>;
}  // namespace RadiationTransport::M1Grey::BoundaryCorrections
