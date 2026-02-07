// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/ControlErrors/Size/DeltaRDriftInward.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>

#include "ControlSystem/ControlErrors/Size/AhSpeed.hpp"
#include "ControlSystem/ControlErrors/Size/DeltaR.hpp"
#include "ControlSystem/ControlErrors/Size/DeltaRDriftOutward.hpp"
#include "ControlSystem/ControlErrors/Size/DeltaRNoDrift.hpp"
#include "Utilities/StdHelpers.hpp"

namespace control_system::size::States {

std::unique_ptr<State> DeltaRDriftInward::get_clone() const {
  return std::make_unique<DeltaRDriftInward>(*this);
}

std::string DeltaRDriftInward::update(
    const gsl::not_null<Info*> info, const StateUpdateArgs& update_args,
    const CrossingTimeInfo& crossing_time_info) const {
  const double Y00 = 0.25 * M_2_SQRTPI;

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

  // spherepack_factor is needed because horizon_00 is a
  // spherepack coefficient, not a spherical harmonic coefficient.
  const double spherepack_factor = sqrt(0.5 * M_PI);

  std::stringstream ss{};

  if (char_speed_is_in_danger) {
    ss << "Current state DeltaRDriftInward. Char speed in danger."
       << " Switching to AhSpeed.\n";
    // Switch to AhSpeed mode. Note that we don't check ComovingCharSpeed
    // like we do in state DeltaR; this behavior agrees with SpEC.

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
    ss << " Target char speed = " << info->target_char_speed << "\n";
    // If the comoving char speed is positive and is not about to
    // cross zero, staying in DeltaRDriftInward mode will rescue the
    // speed automatically (since it drives char speed to comoving
    // char speed, plus a small difference).  But we should decrease
    // the timescale in any case.
    info->suggested_time_scale = crossing_time_info.t_char_speed.value();
    ss << " Suggested timescale = " << info->suggested_time_scale;
  } else if (delta_radius_is_in_danger) {
    info->suggested_time_scale = crossing_time_info.t_delta_radius.value();
    ss << "Current state DeltaRDriftInward. Delta radius in danger. Staying "
          "in DeltaRDriftInward.\n";
    ss << " Suggested timescale = " << info->suggested_time_scale;
  } else if (should_transition_from_state_inward_drift_to_delta_r_no_drift(
                 crossing_time_info.t_drift_limit, info->damping_time,
                 update_args)) {
    ss << "Current state DeltaRDriftInward. Switching to DeltaRNoDrift.\n";
    info->discontinuous_change_has_occurred = true;
    info->state = std::make_unique<States::DeltaRNoDrift>();
    info->suggested_time_scale = crossing_time_info.t_drift_limit;
  } else if (crossing_time_info.t_delta_radius.has_value() and
             info->damping_time >
                 2.0 * spherepack_factor * update_args.horizon_00 * Y00) {
    // Explaination of the above 'if':
    //
    // If crossing_time_info.t_delta_radius has a value, this means
    // that delta_radius is decreasing.  But the entire point of state
    // DeltaRDriftInward is to make delta_radius increase, not
    // decrease.  So if we are in state DeltaRDriftInward and
    // crossing_time_info.t_delta_radius has a value
    // (i.e. delta_radius is decreasing), something is wrong.
    //
    // The thing that is usually wrong is that damping_time is too
    // large, and hence DeltaRDriftInward doesn't have time to make
    // delta_radius increase.  So the fix is to decrease the damping
    // time (a.k.a. suggested_time_scale below).  But we stop
    // decreasing the damping time if it is less than twice the
    // average horizon radius, which is the same criterion SpEC
    // uses. (Here we are assuming that timescales and length scales
    // have the same units, which should be true for horizons).
    ss << "Current state DeltaRDriftInward. RelativeDeltaR is decreasing, "
          "which is probably because timescale is too big (DeltaRDriftInward "
          "should be increasing RelativeDeltaR if control system is working "
          "properly). Decreasing timescale and staying in DeltaRDriftInward.\n";
    // delta_r_drift_inward_decrease_factor is an arbitrary factor
    // chosen by trial and error in SpEC. If this factor is too close
    // to 1, then the timescale does not decrease fast enough.  If
    // this factor is too far from 1, then repeated calls of
    // DeltaRDriftInward::update will decrease the timescale to
    // 2*average_horizon_radius too quickly.
    constexpr double delta_r_drift_inward_decrease_factor = 0.99;
    info->suggested_time_scale =
        info->damping_time * delta_r_drift_inward_decrease_factor;
    info->target_char_speed = target_speed_for_inward_drift(
        update_args.avg_distorted_normal_dot_unit_coord_vector,
        update_args.min_char_speed, update_args.inward_drift_velocity.value());
    ss << " Target char speed = " << info->target_char_speed << "\n";
    ss << " Suggested timescale = " << info->suggested_time_scale;
  } else if (update_args.average_radial_distance.has_value() and
             update_args.average_radial_distance.value() >
                 non_oscillation_drift_inward_factor *
                     update_args.max_allowed_radial_distance.value_or(
                         std::numeric_limits<double>::infinity())) {
    info->discontinuous_change_has_occurred = true;
    ss << "Current state DeltaRDriftInward. We have drifted too far, so "
          "we are switching to DeltaRDriftOutward.\n";
    info->state = std::make_unique<States::DeltaRDriftOutward>();
  } else {
    ss << "Current state DeltaRDriftInward. No change necessary. Staying in "
          "DeltaRDriftInward.";
  }

  return ss.str();
}

double DeltaRDriftInward::control_error(
    const Info& info, const ControlErrorArgs& control_error_args) const {
  // We increase the control error by the target speed, so as to make
  // control_error_delta_r more negative, which gives a negative velocity
  // to delta_r (i.e. a positive velocity to the excision boundary).
  return control_error_args.control_error_delta_r + info.target_char_speed;
}

#if defined(SPECTRE_USE_CHARM)
PUP::able::PUP_ID DeltaRDriftInward::my_PUP_ID = 0;  // NOLINT
#endif                                               // SPECTRE_USE_CHARM

double target_speed_for_inward_drift(
    const double avg_distorted_normal_dot_unit_coord_vector,
    const double min_char_speed, const double inward_drift_velocity) {
  // TargetSpeed should be > 0 (we want DeltaR to increase).  And
  // TargetSpeed must be <
  // min_char_speed/avg_distorted_normal_dot_unit_coord_vector, because
  // going into DriftInward will make min_char_speed decrease by
  // TargetSpeed*avg_distorted_normal_dot_unit_coord_vector. The time
  // it takes v to cross zero (assuming v decreases linearly, only a
  // rough approximation) is
  // Tau*min_char_speed/avg_distorted_normal_dot_unit_coord_vector*TargetSpeed,
  // where Tau is the damping timescale.  Therefore choosing
  // TargetSpeed < fudge *
  // min_char_speed/avg_distorted_normal_dot_unit_coord_vector should make
  // v decrease only by a factor of fudge, and it should make the
  // crossing time fudge*Tau.
  constexpr double fudge = 0.5;
  return std::min(
      inward_drift_velocity,
      fudge * min_char_speed / avg_distorted_normal_dot_unit_coord_vector);
}

bool should_transition_from_state_delta_r_to_inward_drift(
    const std::optional<double>& crossing_time_drift_limit,
    const double damping_time, const StateUpdateArgs& update_args) {
  // This function is called ShouldEnterState3FromState2 in SpEC.
  if (update_args.inward_drift_velocity.has_value() and
      crossing_time_drift_limit.has_value() and
      crossing_time_drift_limit.value() < damping_time) {
    return false;
  }
  return should_activate_inward_drift(update_args);
}

bool should_transition_from_state_inward_drift_to_delta_r_no_drift(
    const std::optional<double>& crossing_time_drift_limit,
    const double damping_time, const StateUpdateArgs& update_args) {
  return (not should_transition_from_state_delta_r_to_inward_drift(
      crossing_time_drift_limit, damping_time, update_args));
}

bool should_activate_inward_drift(const StateUpdateArgs& update_args) {
  // This function is called PreferState3OverState2 in SpEC.

  // This drift factor was chosen in SpEC arbitrarily to be 0.9.
  constexpr double inward_drift_limit_buffer_factor = 0.9;

  // The idea of these variables is to check whether either DeltaR or
  // char speed are close to going above the
  // min_average_radial_distance or min_allowed_char_speed values.  If
  // so, then we don't need state DeltaRDriftInward at the moment.
  // For reference, in SpEC these variables are called
  // "DeltaRAlmostAboveState3Limit" and
  // "CharSpeedAlmostAboveState3Limit".
  const bool delta_r_almost_above_inward_drift_limit =
      update_args.min_allowed_radial_distance.has_value() and
      update_args.average_radial_distance.value() >
          inward_drift_limit_buffer_factor *
              update_args.min_allowed_radial_distance.value();
  const bool char_speed_almost_above_inward_drift_limit =
      update_args.min_allowed_char_speed.has_value() and
      update_args.min_char_speed >
          inward_drift_limit_buffer_factor *
              update_args.min_allowed_char_speed.value();

  return (update_args.inward_drift_velocity.has_value() and
          update_args.comoving_char_speed_increasing_inward and
          (update_args.min_allowed_char_speed.has_value() or
           update_args.min_allowed_radial_distance.has_value()) and
          (not delta_r_almost_above_inward_drift_limit) and
          (not char_speed_almost_above_inward_drift_limit));
}

bool ok_to_return_to_state_deltar(const StateUpdateArgs& update_args) {
  // The purpose of delta_r_large_enough_to_stop_inward_drift and
  // char_speed_large_enough_to_stop_inward_drift below is to stop
  // the scenario in which either the CharSpeed or DeltaR approaches
  // the limit "min_allowed_radial_distance" and the timescale gets
  // cut down to a ridiculously small value.  When that scenario
  // happens, we simply exit state DeltaRNoDrift and go back to DeltaR.
  // These variables are called "DeltaRLargeEnoughToExitState4"
  // and "CharSpeedLargeEnoughToExitState4" in SpEC.
  constexpr double stop_inward_drift_buffer_factor = 0.99;
  const bool delta_r_large_enough_to_stop_inward_drift =
      update_args.min_allowed_radial_distance.has_value() and
      update_args.average_radial_distance.value() >
          stop_inward_drift_buffer_factor *
              update_args.min_allowed_radial_distance.value();
  const bool char_speed_large_enough_to_stop_inward_drift =
      update_args.min_allowed_char_speed.has_value() and
      update_args.min_char_speed >
          stop_inward_drift_buffer_factor *
              update_args.min_allowed_char_speed.value();
  return delta_r_large_enough_to_stop_inward_drift or
         char_speed_large_enough_to_stop_inward_drift;
}
}  // namespace control_system::size::States
