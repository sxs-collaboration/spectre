// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <unordered_set>

#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/Neighbors.hpp"
#include "Domain/OrientationMap.hpp"

namespace TestHelpers {
namespace Limiters {

// Construct a Neighbors object with one neighboring element.
template <size_t VolumeDim>
Neighbors<VolumeDim> make_neighbor_with_id(const size_t id) noexcept {
  return {std::unordered_set<ElementId<VolumeDim>>{ElementId<VolumeDim>(id)},
          OrientationMap<VolumeDim>{}};
}

// Construct an Element with one neighboring element in each direction.
//
// The optional argument specifies directions to external boundaries, i.e.,
// directions where there is no neighboring element.
template <size_t VolumeDim>
Element<VolumeDim> make_element(
    const std::unordered_set<Direction<VolumeDim>>&
        directions_of_external_boundaries = {}) noexcept {
  typename Element<VolumeDim>::Neighbors_t neighbors;
  for (const auto dir : Direction<VolumeDim>::all_directions()) {
    // Element has neighbors in directions with internal boundaries
    if (directions_of_external_boundaries.count(dir) == 0) {
      const size_t index =
          1 + 2 * dir.dimension() + (dir.side() == Side::Upper ? 1 : 0);
      neighbors[dir] = make_neighbor_with_id<VolumeDim>(index);
    }
  }
  return Element<VolumeDim>{ElementId<VolumeDim>{0}, neighbors};
}

}  // namespace Limiters
}  // namespace TestHelpers
