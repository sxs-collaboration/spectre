// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <deque>
#include <memory>
#include <utility>

#include "Domain/Structure/ElementId.hpp"
#include "Parallel/Callback.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/Amr/Actions/CollectDataFromChildren.hpp"

/// \cond
namespace db {
template <typename>
class DataBox;
}  // namespace db
/// \endcond

namespace amr::Actions {
/// \brief Creates a new element in an ArrayAlgorithm whose id is `parent_id`
///
/// \details This action is meant to be initially invoked by
/// amr::Actions::AdjustDomain on the amr::Component.  This action inserts a
/// new element with id `parent_id` in the array referenced by
/// `element_proxy`.  A Parallel::SimpleActionCallback `callback` is passed to
/// the constructor of the new DistributedObject, which will invoke
/// amr::Actions::CollectDataFromChildren on the element with id `child_id`.
///
/// This action does not modify anything in the DataBox
struct CreateParent {
  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables, typename ElementProxy>
  static void apply(
      db::DataBox<DbTagList>& /*box*/,
      Parallel::GlobalCache<Metavariables>& cache, const int /*array_index*/,
      ElementProxy element_proxy,
      ElementId<Metavariables::volume_dim> parent_id,
      const ElementId<Metavariables::volume_dim>& child_id,
      std::deque<ElementId<Metavariables::volume_dim>> sibling_ids_to_collect) {
    auto child_proxy = element_proxy[child_id];
    element_proxy[parent_id].insert(
        cache.thisProxy, Parallel::Phase::AdjustDomain,
        std::make_unique<Parallel::SimpleActionCallback<
            CollectDataFromChildren, decltype(child_proxy),
            ElementId<Metavariables::volume_dim>,
            std::deque<ElementId<Metavariables::volume_dim>>>>(
            child_proxy, std::move(parent_id),
            std::move(sibling_ids_to_collect)));
  }
};
}  // namespace amr::Actions
