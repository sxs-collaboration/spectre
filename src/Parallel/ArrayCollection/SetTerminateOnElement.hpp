// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <mutex>
#include <stdexcept>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Parallel/ArrayCollection/Tags/NumberOfElementsTerminated.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/NodeLock.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace Parallel::Actions {
/// \brief Set the element with ID `my_element_id` to be terminated. This is
/// always invoked on the local component via
/// `Parallel::local_synchronous_action`
///
/// In the future we can use this to try and elide Charm++ calls when the
/// receiver is local.
struct SetTerminateOnElement {
  using return_type = void;

  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables, size_t Dim>
  static return_type apply(
      db::DataBox<DbTagList>& box, gsl::not_null<Parallel::NodeLock*> node_lock,
      gsl::not_null<Parallel::GlobalCache<Metavariables>*> cache,
      const ElementId<Dim>& my_element_id, bool terminate_value);
};

template <typename ParallelComponent, typename DbTagList,
          typename Metavariables, size_t Dim>
auto SetTerminateOnElement::apply(
    db::DataBox<DbTagList>& box,
    const gsl::not_null<Parallel::NodeLock*> node_lock,
    const gsl::not_null<Parallel::GlobalCache<Metavariables>*> cache,
    const ElementId<Dim>& my_element_id, const bool terminate_value)
    -> return_type {
  std::lock_guard node_guard(*node_lock);
  db::mutate<typename ParallelComponent::element_collection_tag,
             Parallel::Tags::NumberOfElementsTerminated>(
      [&cache, &my_element_id, &node_lock, terminate_value](
          const auto element_collection_ptr,
          const auto num_terminated_ptr) -> void {
        try {
          // Note: The nodelock is checked by set_terminate as a sanity
          // check and as a way of making set_terminate difficult to use
          // without this action.
          //
          // Note: since this is a local synchronous action running on the
          // element that called it, we already have the element locked and so
          // setting terminate is threadsafe.
          //
          // Note: the element's set_terminate updates num_terminated
          // correctly so we don't need to handle that here.
          element_collection_ptr->at(my_element_id)
              .set_terminate(num_terminated_ptr, node_lock, terminate_value);

          // Update the state of the nodegroup. Specifically, if _all_
          // elements are set to be terminated, we terminate the nodegroup,
          // otherwise we do not.
          auto* local_branch = Parallel::local_branch(
              Parallel::get_parallel_component<ParallelComponent>(*cache));
          ASSERT(local_branch != nullptr,
                 "The local branch is nullptr, which is inconsistent");
          local_branch->set_terminate(*num_terminated_ptr ==
                                      element_collection_ptr->size());
        } catch (const std::out_of_range&) {
          ERROR("Could not find element with ID " << my_element_id
                                                  << " on node");
        }
      },
      make_not_null(&box));
}
}  // namespace Parallel::Actions
