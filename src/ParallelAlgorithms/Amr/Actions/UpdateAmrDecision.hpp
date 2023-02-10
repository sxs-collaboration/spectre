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
///   * amr::Tags::NeighborFlags
///   * amr::Tags::Flags (if AMR decision is updated)
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
                    const std::array<amr::Flag, Metavariables::volume_dim>&
                        neighbor_amr_flags) {
    constexpr size_t volume_dim = Metavariables::volume_dim;
    auto& my_amr_flags =
        db::get_mutable_reference<amr::Tags::Flags<volume_dim>>(
            make_not_null(&box));
    auto& my_neighbors_amr_flags =
        db::get_mutable_reference<amr::Tags::NeighborFlags<volume_dim>>(
            make_not_null(&box));

    // Actions can be executed in any order.  Therefore we need to check:
    // - If we received flags from a neighbor multiple times, but not in the
    //   order they were sent.  Neighbor flags should only be sent again if
    //   they have changed to a higher priority (i.e. higher integral value of
    //   the flag).
    if (1 == my_neighbors_amr_flags.count(neighbor_id)) {
      const auto& previously_received_flags =
          my_neighbors_amr_flags.at(neighbor_id);
      if (not std::lexicographical_compare(previously_received_flags.begin(),
                                           previously_received_flags.end(),
                                           neighbor_amr_flags.begin(),
                                           neighbor_amr_flags.end())) {
        return;
      }
    }

    my_neighbors_amr_flags.insert_or_assign(neighbor_id, neighbor_amr_flags);

    // Actions can be executed in any order.  Therefore we need to check:
    // - If we have evaluated our own AMR decision.  If not, return.
    if (amr::Flag::Undefined == my_amr_flags[0]) {
      using ::operator<<;
      ASSERT(volume_dim == alg::count(my_amr_flags, amr::Flag::Undefined),
             "Flags should be all Undefined, not " << my_amr_flags);
      return;
    }

    const auto& element = get<::domain::Tags::Element<volume_dim>>(box);

    const bool my_amr_decision_changed = amr::update_amr_decision(
        make_not_null(&my_amr_flags), element, neighbor_id, neighbor_amr_flags);

    if (my_amr_decision_changed) {
      auto& amr_element_array =
          Parallel::get_parallel_component<ParallelComponent>(cache);
      for (const auto& direction_neighbors : element.neighbors()) {
        for (const auto& id : direction_neighbors.second.ids()) {
          Parallel::simple_action<UpdateAmrDecision>(
              amr_element_array[id], element.id(), my_amr_flags);
        }
      }
    }
  }
};
}  // namespace amr::Actions
