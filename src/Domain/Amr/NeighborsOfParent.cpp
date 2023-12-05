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
#include "Domain/Amr/Info.hpp"
#include "Domain/Amr/NewNeighborIds.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace amr {
template <size_t VolumeDim>
std::pair<DirectionMap<VolumeDim, Neighbors<VolumeDim>>,
          DirectionalIdMap<VolumeDim, Mesh<VolumeDim>>>
neighbors_of_parent(
    const ElementId<VolumeDim>& parent_id,
    const std::vector<std::tuple<
        const Element<VolumeDim>&,
        const std::unordered_map<ElementId<VolumeDim>, Info<VolumeDim>>&>>&
        children_elements_and_neighbor_info) {
  std::pair<DirectionMap<VolumeDim, Neighbors<VolumeDim>>,
            DirectionalIdMap<VolumeDim, Mesh<VolumeDim>>>
      result;

  std::vector<ElementId<VolumeDim>> children_ids;
  children_ids.reserve(children_elements_and_neighbor_info.size());
  alg::transform(
      children_elements_and_neighbor_info, std::back_inserter(children_ids),
      [](const auto& child_items) { return std::get<0>(child_items).id(); });

  const auto is_child = [&children_ids](const ElementId<VolumeDim>& id) {
    return alg::find(children_ids, id) != children_ids.end();
  };

  for (const auto& [child, child_neighbor_info] :
       children_elements_and_neighbor_info) {
    for (const auto& [direction, child_neighbors] : child.neighbors()) {
      if (has_potential_sibling(child.id(), direction) and
          is_child(*(child_neighbors.ids().begin()))) {
        continue;  // neighbor in this direction was joined sibling
      }
      const auto new_neighbor_ids_and_meshes = amr::new_neighbor_ids(
          parent_id, direction, child_neighbors, child_neighbor_info);
      std::unordered_set<ElementId<VolumeDim>> new_neighbor_ids;
      for (const auto& [id, mesh] : new_neighbor_ids_and_meshes) {
        result.second.insert_or_assign({direction, id}, mesh);
        new_neighbor_ids.insert(id);
      }
      if (0 == result.first.count(direction)) {
        result.first.emplace(
            direction, Neighbors<VolumeDim>{new_neighbor_ids,
                                            child_neighbors.orientation()});
      } else {
        result.first.at(direction).add_ids(new_neighbor_ids);
      }
    }
  }
  return result;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template std::pair<DirectionMap<DIM(data), Neighbors<DIM(data)>>,           \
                     DirectionalIdMap<DIM(data), Mesh<DIM(data)>>>            \
  neighbors_of_parent(                                                        \
      const ElementId<DIM(data)>& parent_id,                                  \
      const std::vector<std::tuple<                                           \
          const Element<DIM(data)>&,                                          \
          const std::unordered_map<ElementId<DIM(data)>, Info<DIM(data)>>&>>& \
          children_elements_and_neighbor_info);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
}  // namespace amr
