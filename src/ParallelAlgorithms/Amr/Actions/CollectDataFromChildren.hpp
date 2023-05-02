// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <deque>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/Amr/Helpers.hpp"
#include "Domain/Amr/Tags/Flags.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/Amr/Actions/InitializeParent.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace amr::Actions {
/// \brief Collects data from child elements to send to their parent element
/// during adaptive mesh refinement
struct CollectDataFromChildren {
  /// \brief  This function should be called after the parent element has been
  /// created by amr::Actions::CreateParent.
  ///
  /// \details This function sends a copy of all items corresponding to the
  /// mutable_item_creation_tags of `box` of `child_id` to the first sibling in
  /// `sibling_ids_to_collect` by invoking this action.  Finally, the child
  /// element destroys itself.
  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables>
  static void apply(
      db::DataBox<DbTagList>& box, Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<Metavariables::volume_dim>& child_id,
      const ElementId<Metavariables::volume_dim>& parent_id,
      std::deque<ElementId<Metavariables::volume_dim>> sibling_ids_to_collect) {
    std::unordered_map<ElementId<Metavariables::volume_dim>,
                       tuples::tagged_tuple_from_typelist<typename db::DataBox<
                           DbTagList>::mutable_item_creation_tags>>
        children_data{};
    children_data.emplace(
        child_id,
        db::copy_items<
            typename db::DataBox<DbTagList>::mutable_item_creation_tags>(box));
    const auto next_child_id = sibling_ids_to_collect.front();
    sibling_ids_to_collect.pop_front();
    auto& array_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);
    Parallel::simple_action<CollectDataFromChildren>(
        array_proxy[next_child_id], parent_id, sibling_ids_to_collect,
        std::move(children_data));
    array_proxy[child_id].ckDestroy();
  }

  /// \brief  This function should be called after a child element has added its
  /// data to `children_data` by a previous invocation of this action.
  ///
  /// \details This function adds a copy of all items corresponding to the
  /// mutable_item_creation_tags of `box` of `child_id` to `children_data`.
  /// In addition, it checks if there are additional siblings that need to be
  /// added to `sibiling_ids_to_collect`.  (This is necessary as not all
  /// siblings share a face.) If `sibling_ids_to_collect` is not empty, this
  /// action is invoked again on the first sibling in `sibling_ids_to_collect`.
  /// If it is empty, the data is sent to the parent element by calling
  /// amr::Actions::InitializeParent. Finally, the child element destroys
  /// itself.
  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables>
  static void apply(
      db::DataBox<DbTagList>& box, Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<Metavariables::volume_dim>& child_id,
      const ElementId<Metavariables::volume_dim>& parent_id,
      std::deque<ElementId<Metavariables::volume_dim>> sibling_ids_to_collect,
      std::unordered_map<
          ElementId<Metavariables::volume_dim>,
          tuples::tagged_tuple_from_typelist<
              typename db::DataBox<DbTagList>::mutable_item_creation_tags>>
          children_data) {
    constexpr size_t volume_dim = Metavariables::volume_dim;

    // Determine if there are additional siblings that are joining
    // This is necessary because AMR flags are only communicated to face
    // neighbors.
    const auto& element = db::get<::domain::Tags::Element<volume_dim>>(box);
    const auto& my_amr_flags = db::get<amr::Tags::Flags<volume_dim>>(box);
    auto ids_to_join = amr::ids_of_joining_neighbors(element, my_amr_flags);
    for (const auto& id_to_check : ids_to_join) {
      if (alg::count(sibling_ids_to_collect, id_to_check) == 0 and
          children_data.count(id_to_check) == 0) {
        sibling_ids_to_collect.emplace_back(id_to_check);
      }
    }

    children_data.emplace(
        child_id,
        db::copy_items<
            typename db::DataBox<DbTagList>::mutable_item_creation_tags>(box));
    auto& array_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);

    if (sibling_ids_to_collect.empty()) {
      Parallel::simple_action<InitializeParent>(array_proxy[parent_id],
                                                std::move(children_data));
    } else {
      const auto next_child_id = sibling_ids_to_collect.front();
      sibling_ids_to_collect.pop_front();
      Parallel::simple_action<CollectDataFromChildren>(
          array_proxy[next_child_id], parent_id, sibling_ids_to_collect,
          std::move(children_data));
    }
    array_proxy[child_id].ckDestroy();
  }
};
}  // namespace amr::Actions
