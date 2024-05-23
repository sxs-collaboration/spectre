// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <mutex>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DiscontinuousGalerkin/AtomicInboxBoundaryData.hpp"
#include "Parallel/ArrayCollection/ReceiveDataForElement.hpp"
#include "Parallel/ArrayCollection/Tags/ElementLocations.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/NodeLock.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Parallel::Actions {
/*!
 * \brief A local synchronous action where data is communicated to neighbor
 * elements.
 *
 * If the inbox tag type is an `evolution::dg::AtomicInboxBoundaryData` then
 * remote insert for elements on the same node is done in a lock-free manner
 * between the sender and receiver elements, and in a wait-free manner between
 * different sender elements to the same receiver element.
 *
 * The number of messages needed to take the next time step on the receiver
 * element is kept track of and a message is sent to the parallel runtime
 * system (e.g. Charm++) only when the receiver/neighbor element has all the
 * data it needs to take the next time step. This is done so as to reduce
 * pressure on the runtime system by sending fewer messages.
 */
struct SendDataToElement {
  using return_type = void;

  template <typename ParallelComponent, typename DbTagList, size_t Dim,
            typename ReceiveTag, typename ReceiveData, typename Metavariables>
  static return_type apply(
      db::DataBox<DbTagList>& box,
      const gsl::not_null<Parallel::NodeLock*> /*node_lock*/,
      const gsl::not_null<Parallel::GlobalCache<Metavariables>*> cache,
      const ReceiveTag& /*meta*/, const ElementId<Dim>& element_to_execute_on,
      typename ReceiveTag::temporal_id instance, ReceiveData&& receive_data) {
    const size_t my_node = Parallel::my_node<size_t>(*cache);
    auto& element = db::get_mutable_reference<
                        typename ParallelComponent::element_collection_tag>(
                        make_not_null(&box))
                        .at(element_to_execute_on);
    // While we don't mutate the value, we want to avoid locking the DataBox
    // and the nodegroup by using `db::get_mutable_reference`. If/when we
    // start dynamically inserting and removing elements, we'll need to update
    // how we handle this. For example, we might need the containers to have
    // strong stability guarantees.
    const size_t node_of_element =
        db::get_mutable_reference<Parallel::Tags::ElementLocations<Dim>>(
            make_not_null(&box))
            .at(element_to_execute_on);
    auto& my_proxy =
        Parallel::get_parallel_component<ParallelComponent>(*cache);
    if (node_of_element == my_node) {
      [[maybe_unused]] size_t count = 0;
      if constexpr (std::is_same_v<evolution::dg::AtomicInboxBoundaryData<Dim>,
                                   typename ReceiveTag::type>) {
        count = ReceiveTag::insert_into_inbox(
            make_not_null(&tuples::get<ReceiveTag>(element.inboxes())),
            instance, std::forward<ReceiveData>(receive_data));
      } else {
        // Scope so that we minimize how long we lock the inbox.
        std::lock_guard inbox_lock(element.inbox_lock());
        count = ReceiveTag::insert_into_inbox(
            make_not_null(&tuples::get<ReceiveTag>(element.inboxes())),
            instance, std::forward<ReceiveData>(receive_data));
      }
      // A lower bound for the number of neighbors is
      // `2 * Dim - number_of_block_boundaries`, which doesn't give us the
      // exact minimum number of sends we need to do, but gets us close in most
      // cases. If we really wanted to we could also add the number of
      // directions that don't have external boundaries in our neighbors block.
      // if (count >=
      //     (2 * Dim - element_to_execute_on.number_of_block_boundaries())) {
      Parallel::threaded_action<Parallel::Actions::ReceiveDataForElement<>>(
          my_proxy[node_of_element], element_to_execute_on);
      // }
    } else {
      Parallel::threaded_action<Parallel::Actions::ReceiveDataForElement<>>(
          my_proxy[node_of_element], ReceiveTag{}, element_to_execute_on,
          instance, std::forward<ReceiveData>(receive_data));
    }
  }
};
}  // namespace Parallel::Actions
