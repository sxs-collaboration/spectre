// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/AveragedUpwindPenalty.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/UpwindPenalty.hpp"
#include "Utilities/TMPL.hpp"

namespace gh::BoundaryCorrections {
template <size_t Dim>
using standard_boundary_corrections =
    tmpl::list<AveragedUpwindPenalty<Dim>, UpwindPenalty<Dim>>;
}  // namespace gh::BoundaryCorrections
