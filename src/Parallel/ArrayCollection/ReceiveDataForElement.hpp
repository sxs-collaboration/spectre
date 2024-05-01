// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <mutex>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/NodeLock.hpp"
#include "Parallel/Phase.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Parallel::Actions {
/// \brief Receive data for a specific element on the nodegroup.
///
/// If `StartPhase` is `true` then `start_phase(phase)` is called on the
/// `element_to_execute_on`, otherwise `perform_algorithm()` is called on the
/// `element_to_execute_on`.
template <bool StartPhase = false>
struct ReceiveDataForElement {
  /// \brief Entry method called when receiving data from another node.
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex, typename ReceiveData,
            typename ReceiveTag, size_t Dim, typename DistributedObject>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const gsl::not_null<Parallel::NodeLock*> /*node_lock*/,
                    const DistributedObject* /*distributed_object*/,
                    const ReceiveTag& /*meta*/,
                    const ElementId<Dim>& element_to_execute_on,
                    typename ReceiveTag::temporal_id instance,
                    ReceiveData receive_data) {
    ERROR(
        "The multi-node code hasn't been tested. It should work, but be aware "
        "that I haven't tried yet.");
    auto& element_collection = db::get_mutable_reference<
        typename ParallelComponent::element_collection_tag>(
        make_not_null(&box));
    // Note: We'll be able to do a counter-based check here too once that
    // works for LTS in `SendDataToElement`
    ReceiveTag::insert_into_inbox(
        make_not_null(&tuples::get<ReceiveTag>(
            element_collection.at(element_to_execute_on).inboxes())),
        instance, std::move(receive_data));

    apply_impl<ParallelComponent>(cache, element_to_execute_on,
                                  make_not_null(&element_collection));
  }

  /// \brief Entry method call when receiving from same node.
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex, size_t Dim,
            typename DistributedObject>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const gsl::not_null<Parallel::NodeLock*> /*node_lock*/,
                    const DistributedObject* /*distributed_object*/,
                    const ElementId<Dim>& element_to_execute_on) {
    auto& element_collection = db::get_mutable_reference<
        typename ParallelComponent::element_collection_tag>(
        make_not_null(&box));
    apply_impl<ParallelComponent>(cache, element_to_execute_on,
                                  make_not_null(&element_collection));
  }

 private:
  template <typename ParallelComponent, typename Metavariables,
            typename ElementCollection, size_t Dim>
  static void apply_impl(
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<Dim>& element_to_execute_on,
      const gsl::not_null<ElementCollection*> element_collection) {
    const size_t my_node = Parallel::my_node<size_t>(cache);
    auto& my_proxy = Parallel::get_parallel_component<ParallelComponent>(cache);

    if constexpr (StartPhase) {
      const Phase current_phase =
          Parallel::local_branch(
              Parallel::get_parallel_component<ParallelComponent>(cache))
              ->phase();
      auto& element = element_collection->at(element_to_execute_on);
      const std::lock_guard element_lock(element.element_lock());
      element.start_phase(current_phase);
    } else {
      auto& element = element_collection->at(element_to_execute_on);
      std::unique_lock element_lock(element.element_lock(), std::defer_lock);
      if (element_lock.try_lock()) {
        element.perform_algorithm();
      } else {
        Parallel::threaded_action<Parallel::Actions::ReceiveDataForElement<>>(
            my_proxy[my_node], element_to_execute_on);
      }
    }
  }
};
}  // namespace Parallel::Actions
