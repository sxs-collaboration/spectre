// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Amr/NeighborsOfChild.hpp"

#include <array>
#include <cstddef>
#include <unordered_map>
#include <unordered_set>

#include "Domain/Amr/Flag.hpp"
#include "Domain/Amr/Helpers.hpp"
#include "Domain/Amr/NewNeighborIds.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace amr {
template <size_t VolumeDim>
DirectionMap<VolumeDim, Neighbors<VolumeDim>> neighbors_of_child(
    const Element<VolumeDim>& parent,
    const std::array<Flag, VolumeDim>& parent_flags,
    const std::unordered_map<ElementId<VolumeDim>, std::array<Flag, VolumeDim>>&
        parent_neighbor_flags,
    const ElementId<VolumeDim>& child_id) {
  DirectionMap<VolumeDim, Neighbors<VolumeDim>> result;

  const auto sibling_id = [&child_id](const size_t dim) {
    auto sibling_segment_ids = child_id.segment_ids();
    auto& segment_id_to_change = gsl::at(sibling_segment_ids, dim);
    segment_id_to_change = segment_id_to_change.id_of_sibling();
    return ElementId<VolumeDim>{child_id.block_id(), sibling_segment_ids,
                                child_id.grid_index()};
  };

  for (const auto& [direction, old_neighbors] : parent.neighbors()) {
    const auto dim = direction.dimension();
    if (gsl::at(parent_flags, dim) == Flag::Split and
        has_potential_sibling(child_id, direction)) {
      result.emplace(direction, Neighbors<VolumeDim>{{sibling_id(dim)}, {}});
    } else {
      result.emplace(direction,
                     Neighbors<VolumeDim>{
                         new_neighbor_ids(child_id, direction, old_neighbors,
                                          parent_neighbor_flags),
                         old_neighbors.orientation()});
    }
  }

  for (const auto& direction : parent.external_boundaries()) {
    const auto dim = direction.dimension();
    if (gsl::at(parent_flags, dim) == Flag::Split and
        has_potential_sibling(child_id, direction)) {
      result.emplace(direction, Neighbors<VolumeDim>{{sibling_id(dim)}, {}});
    }
  }

  return result;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                 \
  template DirectionMap<DIM(data), Neighbors<DIM(data)>> neighbors_of_child( \
      const Element<DIM(data)>& parent,                                      \
      const std::array<Flag, DIM(data)>& parent_flags,                       \
      const std::unordered_map<ElementId<DIM(data)>,                         \
                               std::array<Flag, DIM(data)>>&                 \
          parent_neighbor_flags,                                             \
      const ElementId<DIM(data)>& child_id);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
}  // namespace amr
