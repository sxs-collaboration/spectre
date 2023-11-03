// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>
#include <pup.h>
#include <string>
#include <type_traits>

#include "ControlSystem/CombinedName.hpp"
#include "ControlSystem/FutureMeasurements.hpp"
#include "ControlSystem/Metafunctions.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "Parallel/ArrayComponentId.hpp"
#include "Parallel/Callback.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct Time;
}  // namespace Tags
namespace control_system::Tags {
template <typename ControlSystems>
struct FutureMeasurements;
struct MeasurementTimescales;
struct Verbosity;
}  // namespace control_system::Tags
/// \endcond

namespace control_system {
/// \ingroup ControlSystemsGroup
/// \ingroup EventsAndTriggersGroup
/// Trigger for control system measurements.
///
/// This trigger is only intended to be used with the
/// `control_system::Event` event.  A specialization of this trigger
/// will be created during control system initialization for each
/// unique \ref control_system::protocols::Measurement "measurement".
///
/// These triggers must be added to the \ref
/// Options::protocols::FactoryCreation "factory_creation" struct in
/// the metavariables, even though they cannot be created from the
/// input file.  The `control_system::control_system_triggers`
/// metafunction provides the list of triggers to include.
template <typename ControlSystems>
class Trigger : public DenseTrigger {
  static_assert(tmpl::size<ControlSystems>::value > 0);
  using measurement = typename tmpl::front<ControlSystems>::measurement;
  static_assert(tmpl::all<ControlSystems,
                          std::is_same<metafunctions::measurement<tmpl::_1>,
                                       tmpl::pin<measurement>>>::value);

 public:
  /// \cond
  // LCOV_EXCL_START
  explicit Trigger(CkMigrateMessage* const msg) : DenseTrigger(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Trigger);  // NOLINT
  // LCOV_EXCL_STOP
  /// \endcond

  // This trigger is created during control system initialization, not
  // from the input file.
  static constexpr bool factory_creatable = false;
  Trigger() = default;

  using is_triggered_return_tags = tmpl::list<>;
  using is_triggered_argument_tags =
      tmpl::list<::Tags::Time,
                 control_system::Tags::FutureMeasurements<ControlSystems>>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  std::optional<bool> is_triggered(
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const Component* /*component*/,
      const double time,
      const control_system::FutureMeasurements& measurement_times) {
    const auto next_measurement = measurement_times.next_measurement();
    ASSERT(next_measurement.has_value(),
           "Checking trigger without knowing next time.");
    const bool triggered = time == *next_measurement;

    if (Parallel::get<Tags::Verbosity>(cache) >= ::Verbosity::Debug) {
      Parallel::printf(
          "%s, time = %.16f: Trigger for control systems (%s) is%s "
          "triggered.\n",
          get_output(array_index), time,
          pretty_type::list_of_names<ControlSystems>(),
          (triggered ? "" : " not"));
    }

    return triggered;
  }

  using next_check_time_return_tags =
      tmpl::list<control_system::Tags::FutureMeasurements<ControlSystems>>;
  using next_check_time_argument_tags = tmpl::list<::Tags::Time>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  std::optional<double> next_check_time(
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const Component* /*component*/,
      const gsl::not_null<control_system::FutureMeasurements*>
          measurement_times,
      const double time) {
    if (measurement_times->next_measurement() == std::optional(time)) {
      measurement_times->pop_front();
    }

    if (not measurement_times->next_measurement().has_value()) {
      const auto& proxy =
          ::Parallel::get_parallel_component<Component>(cache)[array_index];
      const bool is_ready = Parallel::mutable_cache_item_is_ready<
          control_system::Tags::MeasurementTimescales>(
          cache, Parallel::make_array_component_id<Component>(array_index),
          [&](const auto& measurement_timescales) {
            const std::string& measurement_name =
                control_system::combined_name<ControlSystems>();
            ASSERT(measurement_timescales.count(measurement_name) == 1,
                   "Control system trigger expects a measurement timescale "
                   "with the name '"
                       << measurement_name
                       << "' but could not find one. Available names are: "
                       << keys_of(measurement_timescales));
            measurement_times->update(
                *measurement_timescales.at(measurement_name));
            if (not measurement_times->next_measurement().has_value()) {
              return std::unique_ptr<Parallel::Callback>(
                  new Parallel::PerformAlgorithmCallback(proxy));
            }
            return std::unique_ptr<Parallel::Callback>{};
          });

      if (not is_ready) {
        if (Parallel::get<Tags::Verbosity>(cache) >= ::Verbosity::Debug) {
          Parallel::printf(
              "%s, time = %.16f: Trigger for control systems (%s) - Cannot "
              "calculate next_check_time\n",
              get_output(array_index), time,
              pretty_type::list_of_names<ControlSystems>());
        }
        return std::nullopt;
      }
    }

    const double next_trigger = *measurement_times->next_measurement();
    ASSERT(next_trigger > time,
           "Next trigger is in the past: " << next_trigger << " > " << time);

    if (Parallel::get<Tags::Verbosity>(cache) >= ::Verbosity::Debug) {
      Parallel::printf(
          "%s, time = %.16f: Trigger for control systems (%s) - next check "
          "time is %.16f\n",
          get_output(array_index), time,
          pretty_type::list_of_names<ControlSystems>(), next_trigger);
    }

    return {next_trigger};
  }
};

/// \cond
template <typename ControlSystems>
PUP::able::PUP_ID Trigger<ControlSystems>::my_PUP_ID = 0;  // NOLINT
/// \endcond

// This metafunction is tested in Test_EventTriggerMetafunctions.cpp

/// \ingroup ControlSystemGroup
/// The list of triggers needed for measurements for a list of control
/// systems.
template <typename ControlSystems>
using control_system_triggers = tmpl::transform<
    metafunctions::measurements_t<ControlSystems>,
    tmpl::bind<Trigger, metafunctions::control_systems_with_measurement<
                            tmpl::pin<ControlSystems>, tmpl::_1>>>;
}  // namespace control_system
