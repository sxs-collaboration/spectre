
// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <charm++.h>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "Parallel/Callback.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/Amr/Actions/SendDataToChildren.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

/// \cond
namespace db {
template <typename>
class DataBox;
}  // namespace db
/// \endcond

namespace amr::Actions {
/// \brief Creates a new element in an ArrayAlgorithm whose id is `child_id`
///
/// \details This action is meant to be initially invoked by
/// amr::Actions::AdjustDomain on the amr::Component.  This action inserts a new
/// element with id `children_ids[index_of_child_id]` in the array referenced by
/// `element_proxy`.  A Parallel::SimpleActionCallback is passed to the
/// constructor of the new DistributedObject.  If `index_of_child_id` is that of
/// the last element of `children_ids`, the Parallel::SimpleActionCallback will
/// invoke amr::Actions::SendDataToChildren on the element with id `parent_id`.
/// Otherwise, it will invoke amr::Actions::CreateChild on the next element of
/// `children_ids`.
///
/// This action does not modify anything in the DataBox
struct CreateChild {
  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables, typename ElementProxy>
  static void apply(
      db::DataBox<DbTagList>& /*box*/,
      Parallel::GlobalCache<Metavariables>& cache, const int /*array_index*/,
      ElementProxy element_proxy,
      ElementId<Metavariables::volume_dim> parent_id,
      std::vector<ElementId<Metavariables::volume_dim>> children_ids,
      const size_t index_of_child_id) {
    auto my_proxy = Parallel::get_parallel_component<ParallelComponent>(cache);
    const ElementId<Metavariables::volume_dim>& child_id =
        children_ids[index_of_child_id];
    if (index_of_child_id + 1 == children_ids.size()) {
      auto parent_proxy = element_proxy[parent_id];
      element_proxy[child_id].insert(
          cache.thisProxy, Parallel::Phase::AdjustDomain,
          std::make_unique<Parallel::SimpleActionCallback<
              SendDataToChildren, decltype(parent_proxy),
              std::vector<ElementId<Metavariables::volume_dim>>>>(
              parent_proxy, std::move(children_ids)));
    } else {
      element_proxy[child_id].insert(
          cache.thisProxy, Parallel::Phase::AdjustDomain,
          std::make_unique<Parallel::SimpleActionCallback<
              CreateChild, decltype(my_proxy), ElementProxy,
              ElementId<Metavariables::volume_dim>,
              std::vector<ElementId<Metavariables::volume_dim>>, size_t>>(
              my_proxy, std::move(element_proxy), std::move(parent_id),
              std::move(children_ids), index_of_child_id + 1));
    }
  }
};
}  // namespace amr::Actions
