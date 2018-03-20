// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Index.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Side.hpp"

/// \ingroup ComputationalDomainGroup
/// Finds the index in the perpendicular dimension of an element boundary
template <size_t Dim>
size_t index_to_slice_at(const Index<Dim>& extents,
                         const Direction<Dim>& direction) noexcept {
  return direction.side() == Side::Lower ? 0
                                         : extents[direction.dimension()] - 1;
}
