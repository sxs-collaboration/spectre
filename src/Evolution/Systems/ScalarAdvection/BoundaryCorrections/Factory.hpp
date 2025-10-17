// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Evolution/Systems/ScalarAdvection/BoundaryCorrections/Rusanov.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarAdvection::BoundaryCorrections {
template <size_t Dim>
using standard_boundary_corrections = tmpl::list<Rusanov<Dim>>;
}  // namespace ScalarAdvection::BoundaryCorrections
