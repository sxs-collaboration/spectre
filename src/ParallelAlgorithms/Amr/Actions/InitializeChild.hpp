// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/Amr/NeighborsOfChild.hpp"
#include "Domain/Amr/Tags/Flags.hpp"
#include "Domain/Amr/Tags/NeighborFlags.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/ElementRegistration.hpp"
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
/// \brief Initializes the data of a newly created child element from the data
/// of its parent element
///
/// DataBox:
/// - Modifies:
///   * domain::Tags::Element<volume_dim>
///   * domain::Tags::Mesh<volume_dim>
///   * all return_tags of Metavariables::amr::projectors
///
/// \details This action is meant to be invoked by
/// amr::Actions::SendDataToChildren
struct InitializeChild {
  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables, typename... Tags>
  static void apply(db::DataBox<DbTagList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ElementId<Metavariables::volume_dim>& child_id,
                    const tuples::TaggedTuple<Tags...>& parent_items) {
    constexpr size_t volume_dim = Metavariables::volume_dim;
    const auto& parent =
        tuples::get<::domain::Tags::Element<volume_dim>>(parent_items);
    const auto& parent_flags =
        tuples::get<amr::Tags::Flags<volume_dim>>(parent_items);
    const auto& parent_neighbor_flags =
        tuples::get<amr::Tags::NeighborFlags<volume_dim>>(parent_items);
    const auto& parent_mesh =
        tuples::get<::domain::Tags::Mesh<volume_dim>>(parent_items);
    auto neighbors = amr::neighbors_of_child(parent, parent_flags,
                                             parent_neighbor_flags, child_id);
    Element<volume_dim> child(child_id, std::move(neighbors));
    Mesh<volume_dim> child_mesh =
        amr::projectors::mesh(parent_mesh, parent_flags);

    // Default initialization of amr::Tags::Flags and amr::Tags::NeighborFlags
    // is okay
    ::Initialization::mutate_assign<tmpl::list<
        ::domain::Tags::Element<volume_dim>, ::domain::Tags::Mesh<volume_dim>>>(
        make_not_null(&box), std::move(child), std::move(child_mesh));

    tmpl::for_each<typename Metavariables::amr::projectors>(
        [&box, &parent_items](auto projector_v) {
          using projector = typename decltype(projector_v)::type;
          db::mutate_apply<projector>(make_not_null(&box), parent_items);
        });

    Parallel::register_element<ParallelComponent>(box, cache, child_id);
  }
};
}  // namespace amr::Actions
