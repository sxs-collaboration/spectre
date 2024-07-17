// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>

#include "ControlSystem/CombinedName.hpp"
#include "ControlSystem/FutureMeasurements.hpp"
#include "ControlSystem/Metafunctions.hpp"
#include "ControlSystem/Tags/FutureMeasurements.hpp"
#include "ControlSystem/Tags/MeasurementTimescales.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "ControlSystem/Trigger.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/EventsAndDenseTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Time/ChooseLtsStepSize.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct TimeStep;
struct EventsAndDenseTriggers;
}  // namespace Tags
namespace domain::Tags {
struct FunctionsOfTime;
}  // namespace domain::Tags
namespace tuples {
template <typename... Tags>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace control_system::Actions {
/// \ingroup ControlSystemGroup
/// \brief Set up the element component for control-system measurements.
///
/// DataBox changes:
/// - Adds:
///   * `Parallel::Tags::FromGlobalCache<
///     ::control_system::Tags::MeasurementTimescales>`
///
/// - Removes: nothing
/// - Modifies: nothing
template <typename ControlSystems>
struct InitializeMeasurements {
  using control_system_groups =
      tmpl::transform<metafunctions::measurements_t<ControlSystems>,
                      metafunctions::control_systems_with_measurement<
                          tmpl::pin<ControlSystems>, tmpl::_1>>;

  using simple_tags =
      tmpl::transform<control_system_groups,
                      tmpl::bind<Tags::FutureMeasurements, tmpl::_1>>;
  using const_global_cache_tags = tmpl::list<Tags::MeasurementsPerUpdate>;
  using mutable_global_cache_tags =
      tmpl::list<control_system::Tags::MeasurementTimescales>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const double initial_time = db::get<::Tags::Time>(box);
    const int measurements_per_update =
        db::get<Tags::MeasurementsPerUpdate>(box);
    const auto& timescales = Parallel::get<Tags::MeasurementTimescales>(cache);
    tmpl::for_each<control_system_groups>([&](auto group_v) {
      using group = tmpl::type_from<decltype(group_v)>;
      const bool active =
          timescales.at(combined_name<group>())->func(initial_time)[0][0] !=
          std::numeric_limits<double>::infinity();
      db::mutate<Tags::FutureMeasurements<group>>(
          [&](const gsl::not_null<FutureMeasurements*> measurements) {
            if (active) {
              *measurements = FutureMeasurements(
                  static_cast<size_t>(measurements_per_update), initial_time);
            } else {
              *measurements = FutureMeasurements(
                  1, std::numeric_limits<double>::infinity());
            }
          },
          make_not_null(&box));
    });

    db::mutate<::Tags::EventsAndDenseTriggers>(
        [](const gsl::not_null<EventsAndDenseTriggers*>
               events_and_dense_triggers) {
          tmpl::for_each<metafunctions::measurements_t<ControlSystems>>(
              [&events_and_dense_triggers](auto measurement_v) {
                using measurement = tmpl::type_from<decltype(measurement_v)>;
                using control_system_group =
                    metafunctions::control_systems_with_measurement_t<
                        ControlSystems, measurement>;
                using events = tmpl::transform<
                    typename measurement::submeasurements,
                    metafunctions::event_from_submeasurement<
                        tmpl::pin<control_system_group>, tmpl::_1>>;
                std::vector<std::unique_ptr<::Event>> vector_of_events =
                    tmpl::as_pack<events>([](auto... events_v) {
                      return make_vector<std::unique_ptr<::Event>>(
                          std::make_unique<
                              tmpl::type_from<decltype(events_v)>>()...);
                    });
                events_and_dense_triggers->add_trigger_and_events(
                    std::make_unique<
                        control_system::Trigger<control_system_group>>(),
                    std::move(vector_of_events));
              });
        },
        make_not_null(&box));

    // Ensure that the initial time step is small enough that we don't
    // need to perform any measurements to complete it.  This is only
    // necessary for TimeSteppers that use the self-start procedure,
    // because they cannot adjust their step size on the first step
    // after self-start, but it isn't harmful to do it in other cases.
    //
    // Unlike in the steady-state step-limiting code, we don't do
    // anything clever looking at measurement times or planning ahead
    // for future steps.  Avoiding a single non-ideal step isn't worth
    // the added complexity.
    double earliest_expiration = std::numeric_limits<double>::infinity();
    for (const auto& fot :
         Parallel::get<domain::Tags::FunctionsOfTime>(cache)) {
      earliest_expiration =
          std::min(earliest_expiration, fot.second->time_bounds()[1]);
    }
    const auto& time_step = db::get<::Tags::TimeStep>(box);
    const auto start_time = time_step.slab().start();
    if ((start_time + time_step).value() > earliest_expiration) {
      db::mutate<::Tags::TimeStep>(
          [&](const gsl::not_null<TimeDelta*> step) {
            if constexpr (Metavariables::local_time_stepping) {
              *step = choose_lts_step_size(
                  start_time,
                  0.99 * (earliest_expiration - start_time.value()));
            } else {
              *step = Slab(start_time.value(), earliest_expiration).duration();
            }
          },
          make_not_null(&box));
    }

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace control_system::Actions
