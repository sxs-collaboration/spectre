// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/ControlErrors/Size/DeltaRDriftOutward.hpp"

#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>

#include "ControlSystem/ControlErrors/Size/AhSpeed.hpp"
#include "ControlSystem/ControlErrors/Size/DeltaR.hpp"
#include "Utilities/StdHelpers.hpp"

namespace control_system::size::States {

std::unique_ptr<State> DeltaRDriftOutward::get_clone() const {
  return std::make_unique<DeltaRDriftOutward>(*this);
}

std::string DeltaRDriftOutward::update(
    const gsl::not_null<Info*> info, const StateUpdateArgs& update_args,
    const CrossingTimeInfo& crossing_time_info) const {
  // Note that delta_radius_is_in_danger and char_speed_is_in_danger
  // can be different for different States.

  // The value of 0.99 was chosen by trial and error in SpEC.
  // It should be slightly less than unity but nothing should be
  // sensitive to small changes in this value.
  constexpr double time_tolerance_for_delta_r_in_danger = 0.99;
  const bool delta_radius_is_in_danger =
      crossing_time_info.horizon_will_hit_excision_boundary_first and
      crossing_time_info.t_delta_radius.value_or(
          std::numeric_limits<double>::infinity()) <
          info->damping_time * time_tolerance_for_delta_r_in_danger;
  const bool char_speed_is_in_danger =
      crossing_time_info.char_speed_will_hit_zero_first and
      crossing_time_info.t_char_speed.value_or(
          std::numeric_limits<double>::infinity()) < info->damping_time and
      not delta_radius_is_in_danger;

  std::stringstream ss{};

  if (char_speed_is_in_danger) {
    ss << "Current state DeltaRDriftOutward. Char speed in danger."
       << " Switching to AhSpeed.\n";
    // Switch to AhSpeed mode. Note that we don't check ComovingCharSpeed
    // like we do in state DeltaR; this behavior agrees with SpEC.

    // This factor prevents oscillating between states Initial and
    // AhSpeed.  It needs to be slightly greater than unity, but the
    // control system should not be sensitive to the exact
    // value. The value of 1.01 was chosen arbitrarily in SpEC and
    // never needed to be changed.
    constexpr double non_oscillation_factor = 1.01;
    info->discontinuous_change_has_occurred = true;
    info->state = std::make_unique<States::AhSpeed>();
    info->target_char_speed =
        update_args.min_char_speed * non_oscillation_factor;
    ss << " Target char speed = " << info->target_char_speed << "\n";
    // If the comoving char speed is positive and is not about to
    // cross zero, staying in DeltaRDriftOutward mode will rescue the speed
    // automatically (since it drives char speed to comoving char
    // speed).  But we should decrease the timescale in any case.
    info->suggested_time_scale = crossing_time_info.t_char_speed;
    ss << " Suggested timescale = " << info->suggested_time_scale;
  } else if (delta_radius_is_in_danger) {
    info->suggested_time_scale = crossing_time_info.t_delta_radius;
    ss << "Current state DeltaRDriftOutward. Delta radius in danger. Staying "
          "in DeltaRDriftOutward.\n";
    ss << " Suggested timescale = " << info->suggested_time_scale;
  } else if (update_args.average_radial_distance.has_value() and
             update_args.average_radial_distance.value() <
                 update_args.max_allowed_radial_distance.value()) {
    ss << "Current state DeltaRDriftOutward. Switching to DeltaR.";
    info->discontinuous_change_has_occurred = true;
    info->state = std::make_unique<States::DeltaR>();
  } else {
    ss << "Current state DeltaRDriftOutward. No change necessary. Staying in "
          "DeltaRDriftOutward.";
  }

  return ss.str();
}

double DeltaRDriftOutward::control_error(
    const Info& /*info*/, const ControlErrorArgs& control_error_args) const {
  return control_error_args.control_error_delta_r_outward.value_or(
      std::numeric_limits<double>::signaling_NaN());
}

PUP::able::PUP_ID DeltaRDriftOutward::my_PUP_ID = 0;
}  // namespace control_system::size::States
