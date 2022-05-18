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
#include "Parallel/CharmPupable.hpp"
#include "Utilities/Gsl.hpp"
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

  Result is_triggered(
      const double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          measurement_timescales) {
    const double next_measurement = tmpl::as_pack<ControlSystems>(
        [&measurement_timescales, &time](auto... control_systems) {
          return std::min(
              {min(measurement_timescales
                       .at(tmpl::type_from<decltype(control_systems)>::name())
                       ->func(time)[0])...});
        });

    // This will happen if an executable has control systems, but all functions
    // of time were overriden by ones read in from a file. So there is no need
    // to trigger control systems
    if (next_measurement == std::numeric_limits<double>::infinity()) {
      next_trigger_ = next_measurement;
      return {false, *next_trigger_};
    }

    // At least one control system is active
    if (UNLIKELY(not next_trigger_.has_value())) {
      // First call
      next_trigger_ = time;
    }
    const bool triggered = time == *next_trigger_;
    if (triggered) {
      *next_trigger_ += next_measurement;
    }
    return {triggered, *next_trigger_};
  }

  using is_ready_argument_tags = tmpl::list<::Tags::Time>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  static bool is_ready(Parallel::GlobalCache<Metavariables>& cache,
                       const ArrayIndex& array_index,
                       const Component* component, const double time) {
    return tmpl::as_pack<ControlSystems>([&array_index, &cache, &component,
                                          &time](auto... control_systems) {
      return domain::functions_of_time_are_ready<
          control_system::Tags::MeasurementTimescales>(
          cache, array_index, component, time,
          std::array{tmpl::type_from<decltype(control_systems)>::name()...});
    });
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    DenseTrigger::pup(p);
    p | next_trigger_;
  }

 private:
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
