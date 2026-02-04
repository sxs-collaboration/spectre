// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <pup.h>
#include <string>

#include "ControlSystem/ControlErrors/Size/Info.hpp"
#include "ControlSystem/ControlErrors/Size/State.hpp"
#include "Options/String.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace control_system::size::States {
class DeltaRDriftInward : public State {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Controls the velocity of the excision surface to maintain a constant "
      "separation between the excision surface and the horizon surface with a "
      "small inward radial velocity. This is state 3 in SpEC."};
  DeltaRDriftInward() = default;
  std::string name() const override { return "DeltaRDriftInward"; }
  size_t number() const override { return 3; }
  std::unique_ptr<State> get_clone() const override;
  std::string update(gsl::not_null<Info*> info,
                     const StateUpdateArgs& update_args,
                     const CrossingTimeInfo& crossing_time_info) const override;
  /// The return value is Q from Eq. 96 of \cite Hemberger2012jz, plus
  /// an inward velocity term.
  double control_error(
      const Info& info,
      const ControlErrorArgs& control_error_args) const override;

  WRAPPED_PUPable_decl_template(DeltaRDriftInward);  // NOLINT
};

// The following are helper functions that are used in many
// of the states, for transitions to/from DeltaRDriftInward.

/// Value of target_char_speed when state DeltaRDriftInward is in effect.
double target_speed_for_inward_drift(
    double avg_distorted_normal_dot_unit_coord_vector, double min_char_speed,
    double inward_drift_velocity);

/// Returs true if we should transition from state DeltaR to state
/// DeltaRDriftInward.
bool should_transition_from_state_delta_r_to_inward_drift(
    const std::optional<double>& crossing_time_drift_limit, double damping_time,
    const StateUpdateArgs& update_args);

/// Returns true if we should transition from state DeltaRDriftInward
/// to state DeltaRNoDrift.
bool should_transition_from_state_inward_drift_to_delta_r_no_drift(
    const std::optional<double>& crossing_time_drift_limit, double damping_time,
    const StateUpdateArgs& update_args);

/// Returns true if we should transition to DeltaRDriftInward rather than
/// to DeltaR.
bool should_activate_inward_drift(const StateUpdateArgs& update_args);

/// Returns true if either CharSpeed approaches min_allowed_char_speed
/// or DeltaR approaches min_allowed_radial_distance close enough
/// that it would be ok to turn off state DeltaRNoDrift.
bool ok_to_return_to_state_deltar(const StateUpdateArgs& update_args);

}  // namespace control_system::size::States
