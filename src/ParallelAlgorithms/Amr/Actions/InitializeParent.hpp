// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Amr/NeighborsOfParent.hpp"
#include "Domain/Amr/Tags/Flags.hpp"
#include "Domain/Amr/Tags/NeighborFlags.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "ParallelAlgorithms/Amr/Projectors/Mesh.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
/// \endcond

namespace amr::Actions {
/// \brief Initializes the data of a newly created parent element from the data
/// of its children elements
///
/// \warning At the moment, this action only initializes the Element, Mesh,
/// and amr::Flag%s of the parent element.  It does not initialize any data
/// related to evolution or elliptic solves.
///
/// DataBox:
/// - Modifies:
///   * domain::Tags::Element<volume_dim>
///   * domain::Tags::Mesh<volume_dim>
///
/// \details This action is meant to be invoked by
/// amr::Actions::CollectDataFromChildren
struct InitializeParent {
  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables>
  static void apply(
      db::DataBox<DbTagList>& box,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Metavariables::volume_dim>& parent_id,
      std::unordered_map<
          ElementId<Metavariables::volume_dim>,
          tuples::tagged_tuple_from_typelist<
              typename db::DataBox<DbTagList>::mutable_item_creation_tags>>
          children_items) {
    constexpr size_t volume_dim = Metavariables::volume_dim;
    std::vector<
        std::tuple<const Element<volume_dim>&,
                   const std::unordered_map<ElementId<volume_dim>,
                                            std::array<Flag, volume_dim>>&>>
        children_elements_and_neighbor_flags;
    for (const auto& [_, child_items] : children_items) {
      children_elements_and_neighbor_flags.emplace_back(std::forward_as_tuple(
          tuples::get<::domain::Tags::Element<volume_dim>>(child_items),
          tuples::get<amr::Tags::NeighborFlags<volume_dim>>(child_items)));
    }
    auto parent_neighbors = amr::neighbors_of_parent(
        parent_id, children_elements_and_neighbor_flags);
    Element<volume_dim> parent(parent_id, std::move(parent_neighbors));

    std::vector<Mesh<volume_dim>> projected_children_meshes{};
    projected_children_meshes.reserve(children_items.size());
    for (const auto& [child_id, child_items] : children_items) {
      const auto& child_mesh =
          tuples::get<::domain::Tags::Mesh<volume_dim>>(child_items);
      const auto& child_flags =
          tuples::get<amr::Tags::Flags<volume_dim>>(child_items);
      projected_children_meshes.emplace_back(
          amr::projectors::mesh(child_mesh, child_flags));
    }
    Mesh<volume_dim> parent_mesh =
        amr::projectors::parent_mesh(projected_children_meshes);

    // Default initialization of amr::Tags::Flags and amr::Tags::NeighborFlags
    // is okay
    ::Initialization::mutate_assign<tmpl::list<
        ::domain::Tags::Element<volume_dim>, ::domain::Tags::Mesh<volume_dim>>>(
        make_not_null(&box), std::move(parent), std::move(parent_mesh));

    // In the near future, add the capability of updating all data needed for
    // an evolution or elliptic system
  }
};
}  // namespace amr::Actions
