// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Parallel/ElementRegistration.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/Amr/Actions/InitializeChild.hpp"
#include "ParallelAlgorithms/Amr/Tags.hpp"

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
            typename Metavariables, size_t Dim>
  static void apply(db::DataBox<DbTagList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ElementId<Dim>& element_id,
                    const std::vector<ElementId<Dim>>& ids_of_children) {
    auto& array_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);
    for (const auto& child_id : ids_of_children) {
      Parallel::simple_action<amr::Actions::InitializeChild>(
          array_proxy[child_id],
          db::copy_items<
              typename db::DataBox<DbTagList>::mutable_item_creation_tags>(
              box));
    }

    if constexpr (Metavariables::amr::keep_coarse_grids) {
      if (db::get<amr::Tags::MaxCoarseLevels>(box).value_or(
              std::numeric_limits<size_t>::max()) > 0) {
        // Register the children elements in the parent element
        Parallel::deregister_element<ParallelComponent>(box, cache, element_id);
        ::Initialization::mutate_assign<tmpl::list<amr::Tags::ChildIds<Dim>>>(
            make_not_null(&box),
            std::unordered_set<ElementId<Dim>>{ids_of_children.begin(),
                                               ids_of_children.end()});
        Parallel::register_element<ParallelComponent>(box, cache, element_id);
        // Note: we don't run AMR projectors on the parent element because the
        // parent element didn't change, with the exception of
        // `amr::Tags::ChildIds<Dim>`.
        // Reset the AMR flags
        db::mutate<amr::Tags::Info<Dim>, amr::Tags::NeighborInfo<Dim>>(
            [](const gsl::not_null<amr::Info<Dim>*> amr_info,
               const gsl::not_null<
                   std::unordered_map<ElementId<Dim>, amr::Info<Dim>>*>
                   amr_info_of_neighbors) {
              amr_info_of_neighbors->clear();
              for (size_t d = 0; d < Dim; ++d) {
                amr_info->flags[d] = amr::Flag::Undefined;
              }
            },
            make_not_null(&box));
        return;
      }
    }

    // Destroy the parent element
    Parallel::deregister_element<ParallelComponent>(box, cache, element_id);
    array_proxy[element_id].ckDestroy();
  }
};
}  // namespace amr::Actions
