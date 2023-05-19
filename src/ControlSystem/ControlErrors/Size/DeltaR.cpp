// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/ControlErrors/Size/DeltaR.hpp"

#include <memory>

#include "ControlSystem/ControlErrors/Size/AhSpeed.hpp"

namespace control_system::size::States {

std::unique_ptr<State> DeltaR::get_clone() const {
  return std::make_unique<DeltaR>(*this);
}

void DeltaR::update(const gsl::not_null<Info*> info,
                    const StateUpdateArgs& update_args,
                    const CrossingTimeInfo& crossing_time_info) const {
  // If update_args.control_error_delta_r is larger than
  // delta_r_control_error_threshold (and neither char speed nor
  // delta radius is in danger), then the timescale is decreased to
  // keep the control error small. This behavior is similar to what
  // TimecaleTuners do, but is triggered only in some situations. The
  // value of 1e-3 was chosen by trial and error in SpEC but it might
  // be helpful to decrease this value in the future if size control
  // needs to be very tight.
  constexpr double delta_r_control_error_threshold = 1.e-3;

  // Note that delta_radius_is_in_danger and char_speed_is_in_danger
  // can be different for different States.

  // The value of 0.99 was chosen by trial and error in SpEC.
  // It should be slightly less than unity but nothing should be
  // sensitive to small changes in this value.
  constexpr double time_tolerance_for_delta_r_in_danger = 0.99;
  const bool delta_radius_is_in_danger =
      crossing_time_info.horizon_will_hit_excision_boundary_first and
      crossing_time_info.t_delta_radius <
          info->damping_time * time_tolerance_for_delta_r_in_danger;
  const bool char_speed_is_in_danger =
      crossing_time_info.char_speed_will_hit_zero_first and
      crossing_time_info.t_char_speed < info->damping_time and
      not delta_radius_is_in_danger;

  if (char_speed_is_in_danger) {
    if (crossing_time_info.t_comoving_char_speed > 0.0 or
        update_args.min_comoving_char_speed < 0.0) {
      // Comoving char speed is negative or threatening to cross zero, so
      // staying in DeltaR mode will not work.  So switch to AhSpeed mode.

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
    }
    // If the comoving char speed is positive and is not about to
    // cross zero, staying in DeltaR mode will rescue the speed
    // automatically (since it drives char speed to comoving char
    // speed).  But we should decrease the timescale in any case.
    info->suggested_time_scale = crossing_time_info.t_char_speed;
  } else if (delta_radius_is_in_danger) {
    info->suggested_time_scale = crossing_time_info.t_delta_radius;
  } else if (update_args.min_comoving_char_speed > 0.0 and
             std::abs(update_args.control_error_delta_r) >
                 delta_r_control_error_threshold) {
    // delta_r_state_decrease_factor should be slightly less than unity.
    // The value of 0.99 below was chosen arbitrarily in SpEC and never
    // needed to be changed.
    constexpr double delta_r_state_decrease_factor = 0.99;
    info->suggested_time_scale =
        info->damping_time * delta_r_state_decrease_factor;
  }
  // Here is where possible transitions to states DeltaRDriftInward and
  // state DeltaRDriftOutward will go.
}

double DeltaR::control_error(const Info& /*info*/,
                             const ControlErrorArgs& control_error_args) const {
  return control_error_args.control_error_delta_r;
}

PUP::able::PUP_ID DeltaR::my_PUP_ID = 0;
}  // namespace control_system::size::States
