// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <unordered_map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Index.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace observers {

/*!
 * \ingroup ObserversGroup
 * \brief %Actions used by the observer parallel component
 */
namespace Actions {

/// \brief Register a class that will call
/// `observers::ThreadedActions::ContributeVolumeData`.
///
/// Should be invoked on ObserverWriter.
struct RegisterVolumeContributorWithObserverWriter {
 public:
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent, typename... ReductionDatums,
            Requires<sizeof...(DbTags) != 0> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const observers::ObservationId& observation_id,
                    const size_t processing_element) noexcept {
    db::mutate<Tags::VolumeObserversRegistered>(
        make_not_null(&box),
        [&observation_id, &processing_element ](
            const gsl::not_null<std::unordered_map<size_t, std::set<size_t>>*>
                volume_observers_registered) noexcept {
          // Currently the only part of the observation_id that is used is the
          // `observation_type_hash()`. But in the future with load balancing
          // we will use the full observation_id, and elements will need
          // to register and unregister themselves at specific times.
          const size_t hash = observation_id.observation_type_hash();

          if (volume_observers_registered->count(hash) == 0) {
            (*volume_observers_registered)[hash] = std::set<size_t>{};
          }
          // We don't care if we insert the same processing element
          // more than once. We care only about which processing
          // elements have registered.
          volume_observers_registered->at(hash).insert(processing_element);
        });
  }
};

/// \brief Register a class that will call
/// `observers::ThreadedActions::WriteReductionData` or
/// `observers::ThreadedActions::ContributeReductionData`.
///
/// Should be invoked on ObserverWriter.
struct RegisterReductionContributorWithObserverWriter {
 public:
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent, typename... ReductionDatums,
            Requires<sizeof...(DbTags) != 0> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const observers::ObservationId& observation_id,
                    const size_t processing_element_or_node,
                    const bool called_from_other_node = false) noexcept {
    const auto node_id = static_cast<size_t>(Parallel::my_node());
    if (not called_from_other_node) {
      // processing_element_or_node is the processing element of the caller.
      db::mutate<Tags::ReductionObserversRegistered>(
          make_not_null(&box),
          [&cache, &node_id, &observation_id, &processing_element_or_node ](
              const gsl::not_null<std::unordered_map<size_t, std::set<size_t>>*>
                  reduction_observers_registered) noexcept {
            // Currently the only part of the observation_id that is used is the
            // `observation_type_hash()`. But in the future with load balancing
            // we will use the full observation_id, and elements will need
            // to register and unregister themselves at specific times.
            const size_t hash = observation_id.observation_type_hash();

            if (reduction_observers_registered->count(hash) == 0) {
              (*reduction_observers_registered)[hash] = std::set<size_t>{};

              // If this is not node 0, call this Action on node 0 to register
              // the nodes that we expect reductions to be called on.
              if (node_id != 0) {
                Parallel::simple_action<
                    Actions::RegisterReductionContributorWithObserverWriter>(
                    Parallel::get_parallel_component<
                        ObserverWriter<Metavariables>>(cache)[0],
                    observation_id, node_id, true);
              }
            }

            // We don't care if we insert the same processing element
            // more than once. We care only about which processing
            // elements have registered.
            reduction_observers_registered->at(hash).insert(
                processing_element_or_node);
          });
    } else if (called_from_other_node) {
      // processing_element_or_node is the node_id of the caller.

      ASSERT(node_id == 0, "Only node zero, not node "
                               << node_id
                               << ", should be called from another node");

      db::mutate<Tags::ReductionObserversRegisteredNodes>(
          make_not_null(&box),
          [&processing_element_or_node, &observation_id ](
              const gsl::not_null<std::unordered_map<size_t, std::set<size_t>>*>
                  reduction_observers_registered_nodes) noexcept {
            const size_t hash = observation_id.observation_type_hash();

            if (reduction_observers_registered_nodes->count(hash) == 0) {
              (*reduction_observers_registered_nodes)[hash] =
                  std::set<size_t>{};
            }
            reduction_observers_registered_nodes->at(hash).insert(
                processing_element_or_node);
          });
    }
  }
};

/*!
 * \brief Action that is called on the Observer parallel component to register
 * the parallel component that will send the data to the observer
 */
struct RegisterSenderWithSelf {
  template <
      typename DbTagList, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<tmpl::list_contains_v<
                   DbTagList, observers::Tags::ReductionArrayComponentIds> and
               tmpl::list_contains_v<
                   DbTagList, observers::Tags::VolumeArrayComponentIds>> =
          nullptr>
  static void apply(db::DataBox<DbTagList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const observers::ObservationId& observation_id,
                    const observers::ArrayComponentId& component_id,
                    const TypeOfObservation& type_of_observation) noexcept {
    const auto register_reduction =
        [&box, &cache, &component_id, &observation_id ]() noexcept {
      db::mutate<observers::Tags::ReductionArrayComponentIds>(
          make_not_null(&box), [&component_id](
                                   const auto array_component_ids) noexcept {
            ASSERT(array_component_ids->count(component_id) == 0,
                   "Trying to insert a component_id more than once for "
                   "reduction. This means an element is registering itself "
                   "with the observers more than once.");
            array_component_ids->insert(component_id);
          });
      const auto my_proc = static_cast<size_t>(Parallel::my_proc());
      auto& observer_writer =
          *Parallel::get_parallel_component<
               observers::ObserverWriter<Metavariables>>(cache)
               .ckLocalBranch();
      Parallel::simple_action<
          Actions::RegisterReductionContributorWithObserverWriter>(
          observer_writer, observation_id, my_proc);
    };
    const auto register_volume =
        [&box, &cache, &component_id, &observation_id ]() noexcept {
      db::mutate<observers::Tags::VolumeArrayComponentIds>(
          make_not_null(&box), [&component_id](
                                   const auto array_component_ids) noexcept {
            ASSERT(array_component_ids->count(component_id) == 0,
                   "Trying to insert a component_id more than once for "
                   "volume observation. This means an element is registering "
                   "itself with the observers more than once.");
            array_component_ids->insert(component_id);
          });
      const auto my_proc = static_cast<size_t>(Parallel::my_proc());
      auto& observer_writer =
          *Parallel::get_parallel_component<
               observers::ObserverWriter<Metavariables>>(cache)
               .ckLocalBranch();
      Parallel::simple_action<
          Actions::RegisterVolumeContributorWithObserverWriter>(
          observer_writer, observation_id, my_proc);
    };

    switch (type_of_observation) {
      case TypeOfObservation::ReductionAndVolume:
        register_reduction();
        register_volume();
        return;
      case TypeOfObservation::Reduction:
        register_reduction();
        return;
      case TypeOfObservation::Volume:
        register_volume();
        return;
      default:
        ERROR(
            "Registering an unknown TypeOfObservation. Should be one of "
            "'Reduction', 'Volume', or 'ReductionAndVolume'");
    };
  }
};

/*!
 * \brief Registers itself with the local observer parallel component so the
 * observer knows to expect data from this component, and also whether to expect
 * Reduction, Volume, or both Reduction and Volume data observation calls.
 */
template <observers::TypeOfObservation TypeOfObservation>
struct RegisterWithObservers {
  template <typename DbTagList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static void apply(const db::DataBox<DbTagList>& /*box*/,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index, const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const observers::ObservationId& observation_id) noexcept {
    auto& observer =
        *Parallel::get_parallel_component<observers::Observer<Metavariables>>(
             cache)
             .ckLocalBranch();
    // TODO(): We should really loop through all the events, check if there are
    // any reduction observers, and any volume observers, if so, then we
    // register us as having one of:
    // - TypeOfObservation::Reduction
    // - TypeOfObservation::Volume
    // - TypeOfObservation::ReductionAndVolume
    //
    // At that point we won't need the template parameter on the class anymore
    // and everything can be read from an input file.
    Parallel::simple_action<RegisterSenderWithSelf>(
        observer, observation_id,
        observers::ArrayComponentId(
            std::add_pointer_t<ParallelComponent>{nullptr},
            Parallel::ArrayIndex<std::decay_t<ArrayIndex>>{array_index}),
        TypeOfObservation);
  }
};

/*!
 * \brief Registers a singleton with the ObserverWriter.
 *
 * The singleton that observes is expected to call WriteReductionData
 * on node 0. An example of how to do this is
 *```
 * template <typename DbTags, typename Metavariables>
 * static void apply(
 *     const db::DataBox<DbTags>& box,
 *     Parallel::ConstGlobalCache<Metavariables>& cache,
 *     const typename Metavariables::temporal_id::type& temporal_id) noexcept {
 *   auto& proxy = Parallel::get_parallel_component<
 *       observers::ObserverWriter<Metavariables>>(cache);
 *
 *   // We call this on proxy[0] because the 0th element of a NodeGroup is
 *   // always guaranteed to be present.
 *   Parallel::threaded_action<observers::ThreadedActions::WriteReductionData>(
 *       proxy[0],
 *       observers::ObservationId(temporal_id.time(), ObservationType{}),
 *       std::string{"/" + pretty_type::short_name<InterpolationTargetTag>()},
 *       detail::make_legend(TagsToObserve{}),
 *       detail::make_reduction_data(box, temporal_id.time().value(),
 *                                   TagsToObserve{}));
 * }
 *
 *```
 *
 */
struct RegisterSingletonWithObserverWriter {
  template <typename DbTagList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static void apply(const db::DataBox<DbTagList>& /*box*/,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const observers::ObservationId& observation_id) noexcept {
    // The actual value of the processing element doesn't matter
    // here; it is used only to give an ObserverWriter a count of how many
    // times it will be called.
    constexpr size_t fake_processing_element = 0;

    // We call only on node 0; the observation call will occur only
    // on node 0.
    Parallel::simple_action<
        Actions::RegisterReductionContributorWithObserverWriter>(
        Parallel::get_parallel_component<
            observers::ObserverWriter<Metavariables>>(cache)[0],
        observation_id, fake_processing_element);
  }
};
}  // namespace Actions
}  // namespace observers
