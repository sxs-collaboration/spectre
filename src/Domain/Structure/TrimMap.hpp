// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Utilities/Gsl.hpp"

namespace domain {
/*!
 * \brief Remove entries in `map_to_trim` that aren't face neighbors of the
 * `element`
 */
template <size_t Dim, typename T>
void remove_nonexistent_neighbors(
    const gsl::not_null<DirectionalIdMap<Dim, T>*> map_to_trim,
    const Element<Dim>& element) {
  std::array<DirectionalId<Dim>, maximum_number_of_neighbors(Dim)>
      ids_to_remove{};
  size_t ids_index = 0;
  for (const auto& [neighbor_id, mesh] : *map_to_trim) {
    const auto& neighbors = element.neighbors();
    if (const auto neighbors_it = neighbors.find(neighbor_id.direction());
        neighbors_it != neighbors.end()) {
      if (const auto neighbor_it =
              neighbors_it->second.ids().find(neighbor_id.id());
          neighbor_it == neighbors_it->second.ids().end()) {
        gsl::at(ids_to_remove, ids_index) = neighbor_id;
        ++ids_index;
      }
    } else {
      gsl::at(ids_to_remove, ids_index) = neighbor_id;
      ++ids_index;
    }
  }
  for (size_t i = 0; i < ids_index; ++i) {
    map_to_trim->erase(gsl::at(ids_to_remove, i));
  }
}
}  // namespace domain
