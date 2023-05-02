// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Amr/NeighborsOfParent.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "Domain/Amr/Flag.hpp"
#include "Domain/Amr/Helpers.hpp"
#include "Domain/Amr/NewNeighborIds.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace amr {
template <size_t VolumeDim>
DirectionMap<VolumeDim, Neighbors<VolumeDim>> neighbors_of_parent(
    const ElementId<VolumeDim>& parent_id,
    const std::vector<
        std::tuple<const Element<VolumeDim>&,
                   const std::unordered_map<ElementId<VolumeDim>,
                                            std::array<Flag, VolumeDim>>&>>&
        children_elements_and_neighbor_flags) {
  DirectionMap<VolumeDim, Neighbors<VolumeDim>> result;

  std::vector<ElementId<VolumeDim>> children_ids;
  children_ids.reserve(children_elements_and_neighbor_flags.size());
  alg::transform(
      children_elements_and_neighbor_flags, std::back_inserter(children_ids),
      [](const auto& child_items) { return std::get<0>(child_items).id(); });

  const auto is_child = [&children_ids](const ElementId<VolumeDim>& id) {
    return alg::find(children_ids, id) != children_ids.end();
  };

  for (const auto& [child, child_neighbor_flags] :
       children_elements_and_neighbor_flags) {
    for (const auto& [direction, child_neighbors] : child.neighbors()) {
      if (has_potential_sibling(child.id(), direction) and
          is_child(*(child_neighbors.ids().begin()))) {
        continue;  // neighbor in this direction was joined sibling
      }
      if (0 == result.count(direction)) {
        result.emplace(direction, Neighbors<VolumeDim>{
                                      amr::new_neighbor_ids(
                                          parent_id, direction, child_neighbors,
                                          child_neighbor_flags),
                                      child_neighbors.orientation()});
      } else {
        result.at(direction).add_ids(amr::new_neighbor_ids(
            parent_id, direction, child_neighbors, child_neighbor_flags));
      }
    }
  }
  return result;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template DirectionMap<DIM(data), Neighbors<DIM(data)>> neighbors_of_parent(  \
      const ElementId<DIM(data)>& parent_id,                                   \
      const std::vector<                                                       \
          std::tuple<const Element<DIM(data)>&,                                \
                     const std::unordered_map<ElementId<DIM(data)>,            \
                                              std::array<Flag, DIM(data)>>&>>& \
          children_elements_and_neighbor_flags);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
}  // namespace amr
