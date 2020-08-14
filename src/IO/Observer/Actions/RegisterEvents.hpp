// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <unordered_map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "IO/Observer/Actions/ObserverRegistration.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits/CreateHasTypeAlias.hpp"

namespace observers {
namespace detail {
CREATE_HAS_TYPE_ALIAS(observation_registration_tags)
CREATE_HAS_TYPE_ALIAS_V(observation_registration_tags)
}  // namespace detail

/*!
 * \brief Retrieves the observation type and key from an event, if it is an
 * event that needs to register with the observers.
 *
 * The event must define an `observation_registration_tags` type alias that is a
 * `tmpl::list<>` of all the tags from the DataBox needed to construct the
 * `ObservationKey`, and a `get_observation_type_and_key_for_registration`
 * member function if it is to be registered automatically.
 */
template <typename EventRegistrars, typename DbTagsList>
std::optional<
    std::pair<::observers::TypeOfObservation, ::observers::ObservationKey>>
get_registration_observation_type_and_key(
    const Event<EventRegistrars>& event,
    const db::DataBox<DbTagsList>& box) noexcept {
  std::optional<
      std::pair<::observers::TypeOfObservation, ::observers::ObservationKey>>
      result{};
  bool already_registered = false;
  tmpl::for_each<typename Event<EventRegistrars>::creatable_classes>(
      [&already_registered, &box, &event, &result](auto event_type_v) noexcept {
        using EventType = typename decltype(event_type_v)::type;
        if constexpr (detail::has_observation_registration_tags_v<EventType>) {
          // We require that each event for which
          // `has_observation_registration_tags_v` is true be downcastable to
          // only *one* event type. This is checked by the `already_registered`
          // bool and the error message below. We need to downcast because we do
          // not want to impose that every event be registerable with the
          // IO/observer components. `dynamic_cast` returns a `nullptr` in the
          // case that the downcast failed, which we check for to see if we can
          // call the `get_observation_type_and_key_for_registration` function.
          const auto* const derived_class_ptr =
              dynamic_cast<const EventType*>(&event);
          if (derived_class_ptr != nullptr) {
            if (already_registered) {
              ERROR(
                  "Already registered the event by casting down to a "
                  "different Event derived class. This means you have an "
                  "Event where A inherits from B and both A and B define "
                  "get_observation_type_and_id_for_registration. This "
                  "behavior is not supported. Please make a separate Event.");
            }
            already_registered = true;
            result =
                db::apply<typename EventType::observation_registration_tags>(
                    [&derived_class_ptr](const auto&... args) noexcept {
                      return derived_class_ptr
                          ->get_observation_type_and_key_for_registration(
                              args...);
                    },
                    box);
          }
        }
      });
  return result;
}

namespace Actions {
/*!
 * \brief Registers this element of a parallel component with the local
 * `Observer` parallel component for each triggered observation.
 *
 * \details This tells the `Observer` to expect data from this component, as
 * well as whether each observation is a Reduction or Volume observation.
 * Should be added to the phase dependent action list of the components that
 * contribute data for volume and reduction observations.
 *
 * When this struct is used as an action, the `apply` function will perform the
 * registration with observers. However, this struct also offers the static
 * member functions `perform_registration` and `perform_deregistration` that
 * are needed for either registering when an element is added to a core outside
 * of initialization or deregistering when an element is being eliminated from a
 * core. The use of separate functions is necessary to provide an interface
 * usable outside of iterable actions, e.g. in specialized `pup` functions.
 */
struct RegisterEventsWithObservers {
 private:
  template <typename ParallelComponent, typename RegisterOrDeregisterAction,
            typename DbTagList, typename Metavariables, typename ArrayIndex>
  static void register_or_deregister_impl(
      db::DataBox<DbTagList>& box, Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index) noexcept {
    auto& observer =
        *Parallel::get_parallel_component<observers::Observer<Metavariables>>(
             cache)
             .ckLocalBranch();
    std::vector<
        std::pair<observers::TypeOfObservation, observers::ObservationKey>>
        type_of_observation_and_observation_key_pairs;

    const auto& triggers_and_events =
        db::get<::Tags::EventsAndTriggersBase>(box);
    for (const auto& trigger_and_events :
         triggers_and_events.events_and_triggers()) {
      for (const auto& event : trigger_and_events.second) {
        if (auto obs_type_and_obs_key =
                get_registration_observation_type_and_key(*event, box);
            obs_type_and_obs_key.has_value()) {
          type_of_observation_and_observation_key_pairs.push_back(
              *obs_type_and_obs_key);
        }
      }
    }

    for (const auto& [type_of_observation, observation_key] :
         type_of_observation_and_observation_key_pairs) {
      Parallel::simple_action<RegisterOrDeregisterAction>(
          observer, observation_key,
          observers::ArrayComponentId(
              std::add_pointer_t<ParallelComponent>{nullptr},
              Parallel::ArrayIndex<std::decay_t<ArrayIndex>>{array_index}),
          type_of_observation);
    }
  }

 public:
  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables, typename ArrayIndex>
  static void perform_registration(db::DataBox<DbTagList>& box,
                                   Parallel::GlobalCache<Metavariables>& cache,
                                   const ArrayIndex& array_index) noexcept {
    register_or_deregister_impl<ParallelComponent,
                                RegisterContributorWithObserver>(box, cache,
                                                                 array_index);
  }

  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables, typename ArrayIndex>
  static void perform_deregistration(
      db::DataBox<DbTagList>& box, Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index) noexcept {
    register_or_deregister_impl<ParallelComponent,
                                DeregisterContributorWithObserver>(box, cache,
                                                                   array_index);
  }

  template <typename DbTagList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagList>&&> apply(
      db::DataBox<DbTagList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    perform_registration<ParallelComponent>(box, cache, array_index);
    return {std::move(box)};
  }
};
}  // namespace Actions
}  // namespace observers
