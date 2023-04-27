// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "ControlSystem/Metafunctions.hpp"
#include "ControlSystem/Protocols/Measurement.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "Domain/FunctionsOfTime/FunctionsOfTimeAreReady.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace Tags {
struct Time;
}  // namespace Tags
namespace evolution::Tags {
struct PreviousTriggerTime;
}  // namespace evolution::Tags
namespace control_system::Tags {
struct Verbosity;
}  // namespace control_system::Tags
/// \endcond

namespace control_system {
// This Event is tested in Test_Measurement.cpp

/// \ingroup ControlSystemGroup
/// \ingroup EventsAndTriggersGroup
/// Event for running control system measurements.
///
/// This event is only intended to be used with the
/// `control_system::Trigger` trigger.  A specialization of this event
/// will be created during control system initialization for each
/// unique \ref control_system::protocols::Measurement "measurement".
///
/// These events must be added to the \ref
/// Options::protocols::FactoryCreation "factory_creation" struct in
/// the metavariables, even though they cannot be created from the
/// input file.  The `control_system::control_system_events`
/// metafunction provides the list of events to include.
template <typename ControlSystems>
class Event : public ::Event {
  static_assert(tmpl::size<ControlSystems>::value > 0);
  using measurement = typename tmpl::front<ControlSystems>::measurement;
  static_assert(tmpl::all<ControlSystems,
                          std::is_same<metafunctions::measurement<tmpl::_1>,
                                       tmpl::pin<measurement>>>::value);
  static_assert(tt::assert_conforms_to_v<measurement, protocols::Measurement>);

  template <typename ControlSystem>
  using process_measurement_for_control_system =
      typename ControlSystem::process_measurement;

  using callbacks = tmpl::transform<
      ControlSystems,
      tmpl::bind<process_measurement_for_control_system, tmpl::_1>>;

 public:
  /// \cond
  // LCOV_EXCL_START
  explicit Event(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Event);  // NOLINT
  // LCOV_EXCL_STOP
  /// \endcond

  // This event is created during control system initialization, not
  // from the input file.
  static constexpr bool factory_creatable = false;
  Event() = default;

  using compute_tags_for_observation_box = tmpl::list<>;

  using argument_tags = tmpl::list<::Tags::DataBox>;

  template <typename DbTags, typename Metavariables, typename ArrayIndex,
            typename Component>
  void operator()(const db::DataBox<DbTags>& box,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ArrayIndex& array_index,
                  const Component* const component) const {
    const LinkedMessageId<double> measurement_id{
        db::get<::Tags::Time>(box),
        db::get<::evolution::Tags::PreviousTriggerTime>(box)};
    tmpl::for_each<typename measurement::submeasurements>(
        [&array_index, &box, &cache, &component,
         &measurement_id](auto submeasurement) {
          using Submeasurement = tmpl::type_from<decltype(submeasurement)>;
          db::apply<Submeasurement>(box, measurement_id, cache, array_index,
                                    component, ControlSystems{});
          if (Parallel::get<Tags::Verbosity>(cache) >= ::Verbosity::Debug) {
            Parallel::printf(
                "%s, time = %s: Running control system events for measurement "
                "'%s'.\n",
                get_output(array_index), measurement_id,
                pretty_type::name<Submeasurement>());
          }
        });
  }

  using is_ready_argument_tags = tmpl::list<::Tags::Time>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(const double time, Parallel::GlobalCache<Metavariables>& cache,
                const ArrayIndex& array_index,
                const Component* const component) const {
    // This checks all the functions of time, not just those the
    // measurement is being made for.  We need all of them to access
    // coordinate-dependent quantities, which almost all control
    // measurements will need.
    const bool ready =
        domain::functions_of_time_are_ready<domain::Tags::FunctionsOfTime>(
            cache, array_index, component, time);

    if (Parallel::get<Tags::Verbosity>(cache) >= ::Verbosity::Debug) {
      Parallel::printf("%s, time = %.16f: Control system events are%s ready.\n",
                       get_output(array_index), time, (ready ? "" : " not"));
    }

    return ready;
  }

  bool needs_evolved_variables() const override { return true; }
};

/// \cond
template <typename ControlSystems>
PUP::able::PUP_ID Event<ControlSystems>::my_PUP_ID = 0;  // NOLINT
/// \endcond

// This metafunction is tested in Test_EventTriggerMetafunctions.cpp

/// \ingroup ControlSystemGroup
/// The list of events needed for measurements for a list of control
/// systems.
template <typename ControlSystems>
using control_system_events = tmpl::transform<
    metafunctions::measurements_t<ControlSystems>,
    tmpl::bind<Event, metafunctions::control_systems_with_measurement<
                          tmpl::pin<ControlSystems>, tmpl::_1>>>;
}  // namespace control_system
