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
#include "Parallel/Invoke.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace observers {
/*!
 * \ingroup ObserversGroup
 * \brief %Actions used by the observer parallel component
 */
namespace Actions {
/*!
 * \brief Action that is called on the Observer parallel component to register
 * the parallel component that will send the data to the observer
 */
struct RegisterSenderWithSelf {
  template <
      typename DbTagList, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      typename TemporalId,
      Requires<tmpl::list_contains_v<
                   DbTagList, observers::Tags::ReductionArrayComponentIds> and
               tmpl::list_contains_v<
                   DbTagList, observers::Tags::VolumeArrayComponentIds>> =
          nullptr>
  static void apply(db::DataBox<DbTagList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const TemporalId& /*temporal_id*/,
                    const observers::ArrayComponentId& component_id,
                    const TypeOfObservation& type_of_observation) noexcept {
    const auto register_reduction = [&box, &component_id]() noexcept {
      db::mutate<observers::Tags::ReductionArrayComponentIds>(
          make_not_null(&box), [&component_id](
                                   const auto array_component_ids) noexcept {
            ASSERT(array_component_ids->count(component_id) == 0,
                   "Trying to insert a component_id more than once for "
                   "reduction. This means an element is registering itself "
                   "with the observers more than once.");
            array_component_ids->insert(component_id);
          });
    };
    const auto register_volume = [&box, &component_id]() noexcept {
      db::mutate<observers::Tags::VolumeArrayComponentIds>(
          make_not_null(&box), [&component_id](
                                   const auto array_component_ids) noexcept {
            ASSERT(array_component_ids->count(component_id) == 0,
                   "Trying to insert a component_id more than once for "
                   "volume observation. This means an element is registering "
                   "itself with the observers more than once.");
            array_component_ids->insert(component_id);
          });
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
            typename ParallelComponent, typename TemporalId>
  static void apply(const db::DataBox<DbTagList>& /*box*/,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index, const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const TemporalId& temporal_id) noexcept {
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
        observer, temporal_id,
        observers::ArrayComponentId(
            std::add_pointer_t<ParallelComponent>{nullptr},
            Parallel::ArrayIndex<std::decay_t<ArrayIndex>>{array_index}),
        TypeOfObservation);
  }
};
}  // namespace Actions
}  // namespace observers
