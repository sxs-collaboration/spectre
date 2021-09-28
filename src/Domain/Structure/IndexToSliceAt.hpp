// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Index.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Side.hpp"

/// \ingroup ComputationalDomainGroup
/// \brief Finds the index in the perpendicular dimension of an element boundary
///
/// Optionally provide an `offset` to find an index offset from the element
/// boundary.
template <size_t Dim>
size_t index_to_slice_at(const Index<Dim>& extents,
                         const Direction<Dim>& direction,
                         const size_t offset = 0) {
  return direction.side() == Side::Lower
             ? offset
             : extents[direction.dimension()] - 1 - offset;
}
