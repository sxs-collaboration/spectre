// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/ControlErrors/Size/Info.hpp"

#include <pup.h>
#include <pup_stl.h>

#include "ControlSystem/ControlErrors/Size/State.hpp"
#include "Parallel/CharmPupable.hpp"

namespace control_system::size {

void Info::pup(PUP::er& p) {
  p | state;
  p | damping_time;
  p | target_char_speed;
  p | target_drift_velocity;
  p | suggested_time_scale;
  p | discontinuous_change_has_occurred;
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
