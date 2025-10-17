// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Evolution/Systems/NewtonianEuler/BoundaryCorrections/Hll.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryCorrections/Hllc.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryCorrections/Rusanov.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianEuler::BoundaryCorrections {
template <size_t Dim>
using standard_boundary_corrections =
    tmpl::list<Hll<Dim>, Hllc<Dim>, Rusanov<Dim>>;
}  // namespace NewtonianEuler::BoundaryCorrections
