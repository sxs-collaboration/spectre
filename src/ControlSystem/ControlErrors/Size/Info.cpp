// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/ControlErrors/Size/Info.hpp"

#include <optional>
#include <pup.h>
#include <pup_stl.h>

#include "ControlSystem/ControlErrors/Size/State.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace control_system::size {
Info::Info(std::unique_ptr<State> in_state, double in_damping_time,
           double in_target_char_speed, double in_target_drift_velocity,
           std::optional<double> in_suggested_time_scale,
           bool in_discontinuous_change_has_occurred)
    : state(std::move(in_state)),
      damping_time(in_damping_time),
      target_char_speed(in_target_char_speed),
      target_drift_velocity(in_target_drift_velocity),
      suggested_time_scale(in_suggested_time_scale),
      discontinuous_change_has_occurred(in_discontinuous_change_has_occurred) {}

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,-warnings-as-errors)
Info::Info(const Info& rhs) {
  set_all_but_state(rhs);
  state = rhs.state->get_clone();
}

Info& Info::operator=(const Info& rhs) {
  set_all_but_state(rhs);
  state = rhs.state->get_clone();
  return *this;
}

void Info::pup(PUP::er& p) {
  p | state;
  p | damping_time;
  p | target_char_speed;
  p | target_drift_velocity;
  p | suggested_time_scale;
  p | discontinuous_change_has_occurred;
}

void Info::reset() {
  suggested_time_scale = std::nullopt;
  discontinuous_change_has_occurred = false;
  // Currently nothing actually sets this, but we may want to reset it in the
  // future when we add more States
  // target_drift_velocity = 0.0;
}

void Info::set_all_but_state(const Info& info) {
  damping_time = info.damping_time;
  target_char_speed = info.target_char_speed;
  target_drift_velocity = info.target_drift_velocity;
  suggested_time_scale = info.suggested_time_scale;
  discontinuous_change_has_occurred = info.discontinuous_change_has_occurred;
}

CrossingTimeInfo::CrossingTimeInfo(
    const double char_speed_crossing_time,
    const double comoving_char_speed_crossing_time,
    const double delta_radius_crossing_time)
    : t_char_speed(char_speed_crossing_time),
      t_comoving_char_speed(comoving_char_speed_crossing_time),
      t_delta_radius(delta_radius_crossing_time) {
  if (t_char_speed > 0.0) {
    if (t_delta_radius > 0.0 and t_delta_radius <= t_char_speed) {
      horizon_will_hit_excision_boundary_first = true;
    } else {
      char_speed_will_hit_zero_first = true;
    }
  } else if (t_delta_radius > 0.0) {
    horizon_will_hit_excision_boundary_first = true;
  }
}
}  // namespace control_system::size
