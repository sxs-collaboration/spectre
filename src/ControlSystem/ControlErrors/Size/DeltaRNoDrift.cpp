// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/ControlErrors/Size/DeltaRNoDrift.hpp"

#include <memory>
#include <optional>
#include <sstream>
#include <string>

#include "ControlSystem/ControlErrors/Size/AhSpeed.hpp"
#include "ControlSystem/ControlErrors/Size/DeltaR.hpp"
#include "ControlSystem/ControlErrors/Size/DeltaRDriftInward.hpp"
#include "ControlSystem/ControlErrors/Size/DeltaRDriftOutward.hpp"
#include "Utilities/StdHelpers.hpp"

namespace control_system::size::States {

std::unique_ptr<State> DeltaRNoDrift::get_clone() const {
  return std::make_unique<DeltaRNoDrift>(*this);
}

std::string DeltaRNoDrift::update(
    const gsl::not_null<Info*> info, const StateUpdateArgs& update_args,
    const CrossingTimeInfo& crossing_time_info) const {
  // This factor is present in SpEC, and it is used to prevent
  // oscillations between states.  The value was chosen in SpEC, but
  // nothing should be sensitive to small changes in this value as
  // long as it is slightly greater than unity.
  constexpr double non_oscillation_drift_inward_factor = 1.1;

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
    ss << "Current state DeltaRNoDrift. Char speed in danger.";
    if (crossing_time_info.t_comoving_char_speed.has_value() or
        update_args.min_comoving_char_speed < 0.0) {
      // Comoving char speed is negative or threatening to cross zero, so
      // staying in DeltaRNoDrift mode will not work.  So switch to AhSpeed
      // mode.

      // This factor prevents oscillations between
      // DeltaR/DeltaRInward/DeltaRNoDrift/DeltaROutward and AhSpeed.
      // It needs to be slightly greater than unity, but the control
      // system should not be sensitive to the exact value. The value of
      // 1.01 was chosen arbitrarily in SpEC and never needed to be
      // changed.
      constexpr double non_oscillation_factor = 1.01;
      info->discontinuous_change_has_occurred = true;
      info->state = std::make_unique<States::AhSpeed>();
      info->target_char_speed =
          update_args.min_char_speed * non_oscillation_factor;
      ss << " Switching to AhSpeed.\n";
      ss << " Target char speed = " << info->target_char_speed << "\n";
    } else {
      ss << " Staying in DeltaRNoDrift.\n";
    }
    // If the comoving char speed is positive and is not about to
    // cross zero, staying in DeltaRNoDrift mode will rescue the speed
    // automatically (since it drives char speed to comoving char
    // speed).  But we should decrease the timescale in any case.
    info->suggested_time_scale = crossing_time_info.t_char_speed;
    ss << " Suggested timescale = " << info->suggested_time_scale;
  } else if (delta_radius_is_in_danger) {
    info->suggested_time_scale = crossing_time_info.t_delta_radius;
    ss << "Current state DeltaRNoDrift. Delta radius in danger. Staying in "
          "DeltaRNoDrift.\n";
    ss << " Suggested timescale = " << info->suggested_time_scale;
  } else if ((not crossing_time_info.t_drift_limit.has_value()) or
             ok_to_return_to_state_deltar(update_args)) {
    info->state = std::make_unique<States::DeltaR>();
    ss << "Current state DeltaRNoDrift, but safe to exit. "
          "Going to state DeltaR.\n";
  } else if (crossing_time_info.t_drift_limit.value_or(
                 std::numeric_limits<double>::infinity()) <
                 info->damping_time and
             (update_args.min_allowed_char_speed.has_value() or
              update_args.min_allowed_radial_distance.has_value())) {
    info->suggested_time_scale = crossing_time_info.t_drift_limit;
    ss << "Current state DeltaRNoDrift. Inward drift limit in danger. Staying "
          "in DeltaRNoDrift.\n";
    ss << " Suggested timescale = " << info->suggested_time_scale;
  } else if (update_args.average_radial_distance.has_value() and
             update_args.average_radial_distance.value() >
                 non_oscillation_drift_inward_factor *
                     update_args.max_allowed_radial_distance.value_or(
                         std::numeric_limits<double>::infinity())) {
    info->discontinuous_change_has_occurred = true;
    ss << "Current state DeltaRNoDrift. We have drifted too far, so "
          "we are switching to DeltaRDriftOutward.\n";
    info->state = std::make_unique<States::DeltaRDriftOutward>();
  } else {
    ss << "Current state DeltaRNoDrift. No change necessary. Staying in "
          "DeltaRNoDrift.";
  }

  return ss.str();
}

double DeltaRNoDrift::control_error(
    const Info& /*info*/, const ControlErrorArgs& control_error_args) const {
  return control_error_args.control_error_delta_r;
}

#ifndef __CUDA_ARCH__
// cppcoreguidelines-avoid-non-const-global-variables
PUP::able::PUP_ID DeltaRNoDrift::my_PUP_ID = 0; // NOLINT
#endif                                          // __CUDA_ARCH__
}  // namespace control_system::size::States
