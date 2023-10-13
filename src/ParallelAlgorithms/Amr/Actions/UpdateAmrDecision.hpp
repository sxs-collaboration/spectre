// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <unordered_map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/Amr/Tags/Flags.hpp"
#include "Domain/Amr/Tags/NeighborFlags.hpp"
#include "Domain/Amr/UpdateAmrDecision.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/Amr/Projectors/Mesh.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

namespace amr::Actions {
/// \brief Given the AMR decision of a neighboring Element, potentially update
/// the AMR decision of the target Element.
///
/// DataBox:
/// - Uses:
///   * domain::Tags::Element<volume_dim>
/// - Modifies:
///   * amr::Tags::NeighborInfo
///   * amr::Tags::Info (if AMR decision is updated)
///
/// Invokes:
/// - amr::Actions::UpdateAmrDecision on all neighboring Element%s (if AMR
///   decision is updated)
///
/// \details This Element calls amr::update_amr_decision to see if its
/// AMR decision needs to be updated.  If it does, the Element will call
/// amr::Actions::UpdateAmrDecision on its neighbors.
///
struct UpdateAmrDecision {
  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables>
  static void apply(db::DataBox<DbTagList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ElementId<Metavariables::volume_dim>& /*element_id*/,
                    const ElementId<Metavariables::volume_dim>& neighbor_id,
                    const Info<Metavariables::volume_dim>& neighbor_amr_info) {
    constexpr size_t volume_dim = Metavariables::volume_dim;
    auto& my_amr_info = db::get_mutable_reference<amr::Tags::Info<volume_dim>>(
        make_not_null(&box));
    auto& my_neighbors_amr_info =
        db::get_mutable_reference<amr::Tags::NeighborInfo<volume_dim>>(
            make_not_null(&box));

    // Actions can be executed in any order.  Therefore we need to check:
    // - If we received info from a neighbor multiple times, but not in the
    //   order they were sent.  Neighbor info should only be sent again if
    //   the flags have changed to a higher priority (i.e. higher integral value
    //   of the flag) or the new mesh has a larger number of grid points.
    if (1 == my_neighbors_amr_info.count(neighbor_id)) {
      const auto& previously_received_info =
          my_neighbors_amr_info.at(neighbor_id);
      if (previously_received_info.new_mesh.number_of_grid_points() >=
              neighbor_amr_info.new_mesh.number_of_grid_points() and
          not std::lexicographical_compare(
              previously_received_info.flags.begin(),
              previously_received_info.flags.end(),
              neighbor_amr_info.flags.begin(), neighbor_amr_info.flags.end())) {
        return;
      }
    }

    my_neighbors_amr_info.insert_or_assign(neighbor_id, neighbor_amr_info);

    auto& my_amr_flags = my_amr_info.flags;
    // Actions can be executed in any order.  Therefore we need to check:
    // - If we have evaluated our own AMR decision.  If not, return.
    if (amr::Flag::Undefined == my_amr_flags[0]) {
      using ::operator<<;
      ASSERT(volume_dim == alg::count(my_amr_flags, amr::Flag::Undefined),
             "Flags should be all Undefined, not " << my_amr_flags);
      return;
    }

    const auto& element = get<::domain::Tags::Element<volume_dim>>(box);
    const auto my_initial_new_mesh = my_amr_info.new_mesh;

    const bool my_amr_decision_changed =
        amr::update_amr_decision(make_not_null(&my_amr_flags), element,
                                 neighbor_id, neighbor_amr_info.flags);

    auto& my_new_mesh = my_amr_info.new_mesh;
    my_new_mesh =
        amr::projectors::new_mesh(get<::domain::Tags::Mesh<volume_dim>>(box),
                                  my_amr_flags, element, my_neighbors_amr_info);

    if (my_amr_decision_changed or my_new_mesh != my_initial_new_mesh) {
      auto& amr_element_array =
          Parallel::get_parallel_component<ParallelComponent>(cache);
      for (const auto& direction_neighbors : element.neighbors()) {
        for (const auto& id : direction_neighbors.second.ids()) {
          Parallel::simple_action<UpdateAmrDecision>(amr_element_array[id],
                                                     element.id(), my_amr_info);
        }
      }
    }
  }
};
}  // namespace amr::Actions
