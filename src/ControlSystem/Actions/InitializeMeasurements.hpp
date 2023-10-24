// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>

#include "ControlSystem/CombinedName.hpp"
#include "ControlSystem/Event.hpp"
#include "ControlSystem/FutureMeasurements.hpp"
#include "ControlSystem/Metafunctions.hpp"
#include "ControlSystem/Tags/FutureMeasurements.hpp"
#include "ControlSystem/Tags/MeasurementTimescales.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "ControlSystem/Trigger.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "Evolution/EventsAndDenseTriggers/EventsAndDenseTriggers.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace evolution::Tags {
struct EventsAndDenseTriggers;
}  // namespace evolution::Tags
namespace tuples {
template <typename... Tags>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace control_system::Actions {
/// \ingroup ControlSystemsGroup
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

    db::mutate<evolution::Tags::EventsAndDenseTriggers>(
        [](const gsl::not_null<evolution::EventsAndDenseTriggers*>
               events_and_dense_triggers) {
          tmpl::for_each<metafunctions::measurements_t<ControlSystems>>(
              [&events_and_dense_triggers](auto measurement_v) {
                using control_system_group =
                    metafunctions::control_systems_with_measurement_t<
                        ControlSystems,
                        typename tmpl::type_from<decltype(measurement_v)>>;
                events_and_dense_triggers->add_trigger_and_events(
                    std::make_unique<
                        control_system::Trigger<control_system_group>>(),
                    make_vector<std::unique_ptr<::Event>>(
                        std::make_unique<
                            control_system::Event<control_system_group>>()));
              });
        },
        make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace control_system::Actions
