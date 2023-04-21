// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/Amr/Actions/InitializeChild.hpp"

namespace amr::Actions {
/// \brief Sends data from the parent element to its children elements during
/// adaptive mesh refinement
///
/// \details  This action should be called after all children elements have been
/// created by amr::Actions::CreateChild.  This action sends a copy of all items
/// corresponding to the mutable_item_creation_tags of `box` to each of the
/// elements with `ids_of_children`.  Finally, the parent element destroys
/// itself.
struct SendDataToChildren {
  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables>
  static void apply(db::DataBox<DbTagList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ElementId<Metavariables::volume_dim>& element_id,
                    const std::vector<ElementId<Metavariables::volume_dim>>&
                        ids_of_children) {
    auto& array_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);
    for (const auto& child_id : ids_of_children) {
      Parallel::simple_action<amr::Actions::InitializeChild>(
          array_proxy[child_id],
          db::copy_items<
              typename db::DataBox<DbTagList>::mutable_item_creation_tags>(
              box));
    }
    array_proxy[element_id].ckDestroy();
  }
};
}  // namespace amr::Actions
