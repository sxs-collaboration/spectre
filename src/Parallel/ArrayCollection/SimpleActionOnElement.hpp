// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <mutex>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Parallel/ArrayCollection/Tags/ElementLocations.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/NodeLock.hpp"
#include "Utilities/Gsl.hpp"

namespace Parallel::Actions {
/// \brief A threaded action that calls the simple action `SimpleActionToCall`
/// on either the specified element or broadcasts to the nodegroup.
///
/// If `Block` is `true` then the action will wait until the element is free
/// to be operated on. If it is `false` then a message will be sent to the
/// nodegroup to try invoking the element again.
///
/// This is a threaded action intended to be run on the DG nodegroup.
template <typename SimpleActionToCall, bool Block>
struct SimpleActionOnElement {
  /// \brief Invoke the simple action on a single element
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex,
            typename DistributedObject, size_t Dim, typename... Args>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const gsl::not_null<Parallel::NodeLock*> /*node_lock*/,
                    const DistributedObject* /*distributed_object*/,
                    const ElementId<Dim>& element_to_execute_on,
                    Args&&... args) {
    const size_t my_node = Parallel::my_node<size_t>(cache);
    auto& my_proxy = Parallel::get_parallel_component<ParallelComponent>(cache);

    auto& element_collection = db::get_mutable_reference<
        typename ParallelComponent::element_collection_tag>(
        make_not_null(&box));
    auto& element = element_collection.at(element_to_execute_on);
    std::unique_lock element_lock(element.element_lock(), std::defer_lock);
    if constexpr (Block) {
      element_lock.lock();
      element.template simple_action<SimpleActionToCall>(
          std::forward<Args>(args)...);
    } else {
      if (element_lock.try_lock()) {
        element.template simple_action<SimpleActionToCall>(
            std::forward<Args>(args)...);
      } else {
        Parallel::threaded_action<Parallel::Actions::SimpleActionOnElement<
            SimpleActionToCall, Block>>(my_proxy[my_node],
                                        element_to_execute_on,
                                        std::forward<Args>(args)...);
      }
    }
  }

  /// \brief Invoke the simple action on all elements
  ///
  /// \note This loops over all elements and spawns a message for each element.
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex,
            typename DistributedObject, typename... Args>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const gsl::not_null<Parallel::NodeLock*> /*node_lock*/,
                    const DistributedObject* /*distributed_object*/,
                    const Args&... args) {
    auto& my_proxy = Parallel::get_parallel_component<ParallelComponent>(cache);

    auto& element_locations = db::get_mutable_reference<
        Parallel::Tags::ElementLocations<Metavariables::volume_dim>>(
        make_not_null(&box));
    for (const auto& [element_id, node_id] : element_locations) {
      Parallel::threaded_action<
          Parallel::Actions::SimpleActionOnElement<SimpleActionToCall, Block>>(
          my_proxy[node_id], element_id, args...);
    }
  }
};
}  // namespace Parallel::Actions
