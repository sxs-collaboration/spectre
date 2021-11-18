// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <tuple>
#include <utility>

#include "ControlSystem/Event.hpp"
#include "ControlSystem/Metafunctions.hpp"
#include "ControlSystem/Tags/MeasurementTimescales.hpp"
#include "ControlSystem/Trigger.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "Evolution/EventsAndDenseTriggers/EventsAndDenseTriggers.hpp"
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
///
/// \note This action relies on the `SetupDataBox` aggregated initialization
/// mechanism, so `Actions::SetupDataBox` must be present in the
/// `Initialization` phase action list prior to this action.
template <typename ControlSystems>
struct InitializeMeasurements {
  using simple_tags = db::AddSimpleTags<>;
  using compute_tags = tmpl::list<Parallel::Tags::FromGlobalCache<
      ::control_system::Tags::MeasurementTimescales>>;
  using mutable_global_cache_tags =
      tmpl::list<control_system::Tags::MeasurementTimescales>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    db::mutate<evolution::Tags::EventsAndDenseTriggers>(
        make_not_null(&box),
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
        });
    return std::make_tuple(std::move(box));
  }
};
}  // namespace control_system::Actions
