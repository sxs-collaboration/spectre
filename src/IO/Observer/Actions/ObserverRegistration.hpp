// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <set>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Index.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace observers {
/*!
 * \ingroup ObserversGroup
 * \brief %Actions used by the observer parallel component
 */
namespace Actions {
/// \brief Register an `ArrayComponentId` with a specific
/// `ObservationIdRegistrationKey` that will call
/// `observers::ThreadedActions::ContributeVolumeData`.
///
/// Should be invoked on ObserverWriter by the component that will be
/// contributing the data.
struct RegisterVolumeContributorWithObserverWriter {
 public:
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<DbTagsList>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const observers::ObservationKey& observation_key,
                    const ArrayComponentId& id_of_caller) noexcept {
    if constexpr (tmpl::list_contains_v<
                      DbTagsList, Tags::ExpectedContributorsForObservations>) {
      db::mutate<Tags::ExpectedContributorsForObservations>(
          make_not_null(&box),
          [&id_of_caller, &observation_key](
              const gsl::not_null<std::unordered_map<
                  ObservationKey, std::unordered_set<ArrayComponentId>>*>
                  volume_observers_registered) noexcept {
            if (volume_observers_registered->find(observation_key) ==
                volume_observers_registered->end()) {
              (*volume_observers_registered)[observation_key] =
                  std::unordered_set<ArrayComponentId>{};
            }

            if (UNLIKELY(
                    volume_observers_registered->at(observation_key)
                        .find(id_of_caller) !=
                    volume_observers_registered->at(observation_key).end())) {
              ERROR("Trying to insert a Observer component more than once: "
                    << id_of_caller);
            }

            volume_observers_registered->at(observation_key)
                .insert(id_of_caller);
          });
    } else {
      (void)box;
      (void)observation_key;
      (void)id_of_caller;
      ERROR(
          "Could not find tag "
          "observers::Tags::ExpectedContributorsForObservations in the "
          "DataBox.");
    }
  }
};

/// \brief Deregister an `ArrayComponentId` with a specific
/// `ObservationIdRegistrationKey` that will no longer call
/// `observers::ThreadedActions::ContributeVolumeData`
///
/// Should be invoked on ObserverWriter by the component that was previously
/// registered with
/// `observers::Actions::RegisterVolumeContributorWithObserverWriter`
struct DeregisterVolumeContributorWithObserverWriter {
 public:
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<DbTagsList>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const observers::ObservationKey& observation_key,
                    const ArrayComponentId& id_of_caller) noexcept {
    if constexpr (tmpl::list_contains_v<
                      DbTagsList, Tags::ExpectedContributorsForObservations>) {
      db::mutate<Tags::ExpectedContributorsForObservations>(
          make_not_null(&box),
          [&id_of_caller, &observation_key](
              const gsl::not_null<std::unordered_map<
                  ObservationKey, std::unordered_set<ArrayComponentId>>*>
                  volume_observers_registered) noexcept {
            if (UNLIKELY(volume_observers_registered->find(observation_key) ==
                         volume_observers_registered->end())) {
              ERROR(
                  "Trying to deregister a component associated with an "
                  "unregistered observation key: "
                  << observation_key);
            }

            if (UNLIKELY(
                    volume_observers_registered->at(observation_key)
                        .find(id_of_caller) ==
                    volume_observers_registered->at(observation_key).end())) {
              ERROR("Trying to deregister an unregistered component: "
                    << id_of_caller);
            }

            volume_observers_registered->at(observation_key)
                .erase(id_of_caller);
            if (UNLIKELY(
                    volume_observers_registered->at(observation_key).size() ==
                    0)) {
              volume_observers_registered->erase(observation_key);
            }
          });
    } else {
      (void)box;
      (void)observation_key;
      (void)id_of_caller;
      ERROR(
          "Could not find tag "
          "observers::Tags::ExpectedContributorsForObservations in the "
          "DataBox.");
    }
  }
};

/*!
 * \brief Register a node with the node that writes the reduction data to disk.
 */
struct RegisterReductionNodeWithWritingNode {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const observers::ObservationKey& observation_key,
                    const size_t caller_node_id) noexcept {
    if constexpr (tmpl::list_contains_v<
                      DbTagsList, Tags::NodesExpectedToContributeReductions>) {
      auto& my_proxy =
          Parallel::get_parallel_component<ParallelComponent>(cache);
      const auto node_id = static_cast<size_t>(
          Parallel::my_node(*Parallel::local_branch(my_proxy)));
      ASSERT(node_id == 0, "Only node zero, not node "
                               << node_id
                               << ", should be called from another node");

      db::mutate<Tags::NodesExpectedToContributeReductions>(
          make_not_null(&box),
          [&caller_node_id, &observation_key](
              const gsl::not_null<
                  std::unordered_map<ObservationKey, std::set<size_t>>*>
                  reduction_observers_registered_nodes) noexcept {
            if (reduction_observers_registered_nodes->find(observation_key) ==
                reduction_observers_registered_nodes->end()) {
              (*reduction_observers_registered_nodes)[observation_key] =
                  std::set<size_t>{};
            }
            auto& registered_nodes_for_key =
                reduction_observers_registered_nodes->at(observation_key);
            if (UNLIKELY(registered_nodes_for_key.find(caller_node_id) !=
                         registered_nodes_for_key.end())) {
              ERROR("Already registered node "
                    << caller_node_id << " for reduction observations.");
            }
            registered_nodes_for_key.insert(caller_node_id);
          });
    } else {
      (void)box;
      (void)observation_key;
      (void)caller_node_id;
      ERROR(
          "Do not have tag "
          "observers::Tags::NodesExpectedToContributeReductions "
          "in the DataBox. This means components are registering for "
          "reductions before initialization is complete.");
    }
  }
};

/*!
 * \brief Deregister a node with the node that writes the reduction data to
 * disk.
 */
struct DeregisterReductionNodeWithWritingNode {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const observers::ObservationKey& observation_key,
                    const size_t caller_node_id) noexcept {
    if constexpr (tmpl::list_contains_v<
                      DbTagsList, Tags::NodesExpectedToContributeReductions>) {
      auto& my_proxy =
          Parallel::get_parallel_component<ParallelComponent>(cache);
      const auto node_id = static_cast<size_t>(
          Parallel::my_node(*Parallel::local_branch(my_proxy)));
      ASSERT(node_id == 0,
             "Only node zero, not node "
                 << node_id
                 << " should deregister other nodes in the reduction");

      db::mutate<Tags::NodesExpectedToContributeReductions>(
          make_not_null(&box),
          [&caller_node_id, &observation_key](
              const gsl::not_null<
                  std::unordered_map<ObservationKey, std::set<size_t>>*>
                  reduction_observers_registered_nodes) noexcept {
            if (UNLIKELY(reduction_observers_registered_nodes->find(
                             observation_key) ==
                         reduction_observers_registered_nodes->end())) {
              ERROR(
                  "Trying to deregister a node associated with an unregistered "
                  "observation key: "
                  << observation_key);
            }
            auto& registered_nodes_for_key =
                reduction_observers_registered_nodes->at(observation_key);
            if (UNLIKELY(registered_nodes_for_key.find(caller_node_id) ==
                         registered_nodes_for_key.end())) {
              ERROR("Trying to deregister an unregistered node: "
                    << caller_node_id);
            }
            registered_nodes_for_key.erase(caller_node_id);
            if (UNLIKELY(registered_nodes_for_key.size() == 0)) {
              reduction_observers_registered_nodes->erase(observation_key);
            }
          });
    } else {
      (void)box;
      (void)observation_key;
      (void)caller_node_id;
      ERROR(
          "Do not have tag "
          "observers::Tags::NodesExpectedToContributeReductions "
          "in the DataBox. This means components are registering for "
          "reductions before initialization is complete.");
    }
  }
};

/// \brief Register an `ArrayComponentId` that will call
/// `observers::ThreadedActions::WriteReductionData` or
/// `observers::ThreadedActions::ContributeReductionData` for a specific
/// `ObservationIdRegistrationKey`
///
/// Should be invoked on ObserverWriter by the component that will be
/// contributing the data.
struct RegisterReductionContributorWithObserverWriter {
 public:
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const observers::ObservationKey& observation_key,
                    const ArrayComponentId& id_of_caller) noexcept {
    if constexpr (tmpl::list_contains_v<
                      DbTagsList, Tags::ExpectedContributorsForObservations>) {
      auto& my_proxy =
          Parallel::get_parallel_component<ParallelComponent>(cache);
      const auto node_id = static_cast<size_t>(
          Parallel::my_node(*Parallel::local_branch(my_proxy)));
      db::mutate<Tags::ExpectedContributorsForObservations>(
          make_not_null(&box),
          [&cache, &id_of_caller, &node_id, &observation_key](
              const gsl::not_null<std::unordered_map<
                  ObservationKey, std::unordered_set<ArrayComponentId>>*>
                  reduction_observers_registered) noexcept {
            if (reduction_observers_registered->find(observation_key) ==
                reduction_observers_registered->end()) {
              (*reduction_observers_registered)[observation_key] =
                  std::unordered_set<ArrayComponentId>{};
              Parallel::simple_action<
                  Actions::RegisterReductionNodeWithWritingNode>(
                  Parallel::get_parallel_component<
                      ObserverWriter<Metavariables>>(cache)[0],
                  observation_key, node_id);
            }

            if (LIKELY(reduction_observers_registered->at(observation_key)
                           .find(id_of_caller) ==
                       reduction_observers_registered->at(observation_key)
                           .end())) {
              reduction_observers_registered->at(observation_key)
                  .insert(id_of_caller);
            } else {
              ERROR("Trying to insert a Observer component more than once: "
                    << id_of_caller
                    << " with observation key: " << observation_key);
            }
          });
    } else {
      (void)box;
      (void)cache;
      (void)observation_key;
      (void)id_of_caller;
      ERROR(
          "Could not find tag "
          "observers::Tags::ExpectedContributorsForObservations in the "
          "DataBox.");
    }
  }
};

/// \brief Deregister an `ArrayComponentId` that will no longer call
/// `observers::ThreadedActions::WriteReductionData` or
/// `observers::ThreadedActions::ContributeReductionData` for a specific
/// `ObservationIdRegistrationKey`
///
/// Should be invoked on ObserverWriter by the component that was previously
/// registered by
/// `observers::Actions::RegisterReductionContributorWithObserverWriter`.
struct DeregisterReductionContributorWithObserverWriter {
 public:
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const observers::ObservationKey& observation_key,
                    const ArrayComponentId& id_of_caller) noexcept {
    if constexpr (tmpl::list_contains_v<
                      DbTagsList, Tags::ExpectedContributorsForObservations>) {
      auto& my_proxy =
          Parallel::get_parallel_component<ParallelComponent>(cache);
      const auto node_id = static_cast<size_t>(
          Parallel::my_node(*Parallel::local_branch(my_proxy)));
      db::mutate<Tags::ExpectedContributorsForObservations>(
          make_not_null(&box),
          [&cache, &id_of_caller, &node_id, &observation_key](
              const gsl::not_null<std::unordered_map<
                  ObservationKey, std::unordered_set<ArrayComponentId>>*>
                  reduction_observers_registered) noexcept {
            if (UNLIKELY(
                    reduction_observers_registered->find(observation_key) ==
                    reduction_observers_registered->end())) {
              ERROR(
                  "Trying to deregister a component associated with an "
                  "unregistered observation key: "
                  << observation_key);
            }
            auto& contributors_for_key =
                reduction_observers_registered->at(observation_key);
            if (UNLIKELY(contributors_for_key.find(id_of_caller) ==
                         contributors_for_key.end())) {
              ERROR("Trying to deregister an unregistered component: "
                    << id_of_caller
                    << " with observation key: " << observation_key);
            }
            contributors_for_key.erase(id_of_caller);
            if (UNLIKELY(contributors_for_key.size() == 0)) {
              Parallel::simple_action<
                  Actions::DeregisterReductionNodeWithWritingNode>(
                  Parallel::get_parallel_component<
                      ObserverWriter<Metavariables>>(cache)[0],
                  observation_key, node_id);
              reduction_observers_registered->erase(observation_key);
            }
          });
    } else {
      (void)box;
      (void)cache;
      (void)observation_key;
      (void)id_of_caller;
      ERROR(
          "Could not find tag "
          "observers::Tags::ExpectedContributorsForObservations in the "
          "DataBox.");
    }
  }
};

/*!
 * \brief Register the `ArrayComponentId` that will send the data to the
 * observer for the given `ObservationIdRegistrationKey`
 *
 * Should be invoked on the `Observer` by the contributing component.
 */
struct RegisterContributorWithObserver {
  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<DbTagList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index,
                    const observers::ObservationKey& observation_key,
                    const observers::ArrayComponentId& component_id,
                    const TypeOfObservation& type_of_observation) noexcept {
    if constexpr (tmpl::list_contains_v<
                      DbTagList,
                      observers::Tags::ExpectedContributorsForObservations>) {
      bool observation_key_already_registered = true;
      db::mutate<observers::Tags::ExpectedContributorsForObservations>(
          make_not_null(&box),
          [&component_id, &observation_key,
           &observation_key_already_registered](
              const gsl::not_null<std::unordered_map<
                  ObservationKey, std::unordered_set<ArrayComponentId>>*>
                  array_component_ids) noexcept {
            observation_key_already_registered =
                (array_component_ids->find(observation_key) !=
                 array_component_ids->end());
            if (UNLIKELY(observation_key_already_registered and
                         array_component_ids->at(observation_key)
                                 .find(component_id) !=
                             array_component_ids->at(observation_key).end())) {
              ERROR(
                  "Trying to insert a component_id more than once for "
                  "observation. This means an element is registering itself "
                  "with the observers more than once. The component_id is "
                  << component_id << " and the observation key is "
                  << observation_key);
            }
            array_component_ids->operator[](observation_key)
                .insert(component_id);
          });

      if (observation_key_already_registered) {
        // We only need to register with the observer writer on the first call
        // of this action. Later calls will have already been registered.
        return;
      }

      auto& observer_writer = *Parallel::local_branch(
          Parallel::get_parallel_component<
              observers::ObserverWriter<Metavariables>>(cache));

      switch (type_of_observation) {
        case TypeOfObservation::Reduction:
          Parallel::simple_action<
              Actions::RegisterReductionContributorWithObserverWriter>(
              observer_writer, observation_key,
              ArrayComponentId{std::add_pointer_t<ParallelComponent>{nullptr},
                               Parallel::ArrayIndex<ArrayIndex>(array_index)});
          return;
        case TypeOfObservation::Volume:
          Parallel::simple_action<
              Actions::RegisterVolumeContributorWithObserverWriter>(
              observer_writer, observation_key,
              ArrayComponentId{std::add_pointer_t<ParallelComponent>{nullptr},
                               Parallel::ArrayIndex<ArrayIndex>(array_index)});
          return;
        default:
          ERROR(
              "Registering an unknown TypeOfObservation. Should be one of "
              "'Reduction' or 'Volume'");
      };
    } else {
      ERROR(
          "The DataBox must contain the tag "
          "observers::Tags::ExpectedContributorsForObservations when the "
          "action "
          "RegisterContributorWithObserver is called.");
    }
  }
};

/*!
 * \brief Deregister the `ArrayComponentId` that will no longer send the data to
 * the observer for the given `ObservationIdRegistrationKey`
 *
 * Should be invoked on the `Observer` by the component that was previously
 * registered with `observers::Actions::RegisterContributorWithObserver`.
 */
struct DeregisterContributorWithObserver {
  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<DbTagList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index,
                    const observers::ObservationKey& observation_key,
                    const observers::ArrayComponentId& component_id,
                    const TypeOfObservation& type_of_observation) noexcept {
    if constexpr (tmpl::list_contains_v<
                      DbTagList,
                      observers::Tags::ExpectedContributorsForObservations>) {
      bool all_array_components_have_been_deregistered = false;
      db::mutate<observers::Tags::ExpectedContributorsForObservations>(
          make_not_null(&box),
          [&component_id, &observation_key,
           &all_array_components_have_been_deregistered](
              const gsl::not_null<std::unordered_map<
                  ObservationKey, std::unordered_set<ArrayComponentId>>*>
                  array_component_ids) noexcept {
            if (UNLIKELY(array_component_ids->find(observation_key) ==
                         array_component_ids->end())) {
              ERROR(
                  "Trying to deregister a component associated with an "
                  "unregistered observation key: "
                  << observation_key);
            }
            auto& component_ids_for_key =
                array_component_ids->at(observation_key);
            if (UNLIKELY(component_ids_for_key.find(component_id) ==
                         array_component_ids->at(observation_key).end())) {
              ERROR("Trying to deregister an unregistered component: "
                    << component_id
                    << " with observation key: " << observation_key);
            }
            component_ids_for_key.erase(component_id);
            if (UNLIKELY(component_ids_for_key.size() == 0)) {
              array_component_ids->erase(observation_key);
              all_array_components_have_been_deregistered = true;
            }
          });

      if (not all_array_components_have_been_deregistered) {
        // Only deregister with the observer writer if this deregistration
        // removes the last component for the provided `observation_key`.
        return;
      }

      auto& observer_writer = *Parallel::local_branch(
          Parallel::get_parallel_component<
              observers::ObserverWriter<Metavariables>>(cache));

      switch (type_of_observation) {
        case TypeOfObservation::Reduction:
          Parallel::simple_action<
              Actions::DeregisterReductionContributorWithObserverWriter>(
              observer_writer, observation_key,
              ArrayComponentId{std::add_pointer_t<ParallelComponent>{nullptr},
                               Parallel::ArrayIndex<ArrayIndex>(array_index)});
          return;
        case TypeOfObservation::Volume:
          Parallel::simple_action<
              Actions::DeregisterVolumeContributorWithObserverWriter>(
              observer_writer, observation_key,
              ArrayComponentId{std::add_pointer_t<ParallelComponent>{nullptr},
                               Parallel::ArrayIndex<ArrayIndex>(array_index)});
          return;
        default:
          ERROR(
              "Attempting to deregister an unknown TypeOfObservation. "
              "Should be one of 'Reduction' or 'Volume'");
      };
    } else {
      ERROR(
          "The DataBox must contain the tag "
          "observers::Tags::ExpectedContributorsForObservations when the "
          "action DeregisterContributorWithObserver is called.");
    }
  }
};
}  // namespace Actions
}  // namespace observers
