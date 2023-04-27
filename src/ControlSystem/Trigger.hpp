// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "ControlSystem/Metafunctions.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/FunctionsOfTimeAreReady.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
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
namespace control_system::Tags {
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

  using is_triggered_argument_tags =
      tmpl::list<::Tags::Time, control_system::Tags::MeasurementTimescales>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  std::optional<bool> is_triggered(
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const Component* /*component*/,
      const double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          measurement_timescales) {
    if (UNLIKELY(not next_trigger_.has_value())) {
      // First call

      // This will happen if an executable has control systems, but
      // all functions of time were overriden by ones read in from a
      // file. So there is no need to trigger control systems.  Since
      // we only enter this branch on the first call to the trigger,
      // this is the initial time so we can assume the
      // measurement_timescales are ready.
      if (next_measurement(time, measurement_timescales) ==
          std::numeric_limits<double>::infinity()) {
        next_trigger_ = std::numeric_limits<double>::infinity();
      } else {
        next_trigger_ = time;
      }
    }

    if (Parallel::get<Tags::Verbosity>(cache) >= ::Verbosity::Debug) {
      Parallel::printf(
          "%s, time = %.16f: Trigger for control systems (%s) is%s "
          "triggered.\n",
          get_output(array_index), time,
          pretty_type::list_of_names<ControlSystems>(),
          (time == *next_trigger_ ? "" : " not"));
    }

    return time == *next_trigger_;
  }

  using next_check_time_argument_tags =
      tmpl::list<::Tags::Time, control_system::Tags::MeasurementTimescales>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  std::optional<double> next_check_time(
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const Component* component,
      const double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          measurement_timescales) {
    // At least one control system is active
    const bool is_ready = tmpl::as_pack<ControlSystems>(
        [&array_index, &cache, &component, &time](auto... control_systems) {
          return domain::functions_of_time_are_ready<
              control_system::Tags::MeasurementTimescales>(
              cache, array_index, component, time,
              std::array{
                  tmpl::type_from<decltype(control_systems)>::name()...});
        });
    if (not is_ready) {
      if (Parallel::get<Tags::Verbosity>(cache) >= ::Verbosity::Debug) {
        Parallel::printf(
            "%s, time = %.16f: Trigger - Cannot calculate next_check_time\n",
            get_output(array_index), time);
      }
      return std::nullopt;
    }

    const bool triggered = time == *next_trigger_;
    if (triggered) {
      *next_trigger_ = next_measurement(time, measurement_timescales);
    }

    if (Parallel::get<Tags::Verbosity>(cache) >= ::Verbosity::Debug) {
      Parallel::printf("%s, time = %.16f: Trigger - next check time is %.16f\n",
                       get_output(array_index), time, *next_trigger_);
    }
    return *next_trigger_;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    DenseTrigger::pup(p);
    p | next_trigger_;
  }

 private:
  double next_measurement(
      const double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          measurement_timescales) {
    const double min_measure_time = tmpl::as_pack<ControlSystems>(
        [&measurement_timescales, &time](auto... control_systems) {
          // LCOV_EXCL_START
          return std::min(
              {min(measurement_timescales
                       .at(tmpl::type_from<decltype(control_systems)>::name())
                       ->func(time)[0])...});
          // LCOV_EXCL_STOP
        });

    if (min_measure_time == std::numeric_limits<double>::infinity()) {
      return min_measure_time;
    }

    return time + min_measure_time;
  }

  std::optional<double> next_trigger_{};
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
