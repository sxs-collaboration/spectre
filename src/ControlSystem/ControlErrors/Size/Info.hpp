// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>
#include <pup.h>

/// \cond
namespace control_system::size {
struct State;
}  // namespace control_system::size
/// \endcond

namespace control_system::size {

/// Holds information that is saved between calls of SizeControl.
struct Info {
  Info() = default;
  Info(const Info& rhs);
  Info& operator=(const Info& rhs);
  Info(Info&& rhs) = default;
  Info& operator=(Info&& rhs) = default;

  Info(std::unique_ptr<State> in_state, double in_damping_time,
       double in_target_char_speed, double in_target_drift_velocity,
       std::optional<double> in_suggested_time_scale,
       bool in_discontinuous_change_has_occurred);

  // Info needs to be serializable because it will be
  // stored inside of a ControlError.
  void pup(PUP::er& p);

  /// The current state of size control.
  std::unique_ptr<State> state;
  /// The current damping time associated with size control.
  double damping_time;
  /// target_char_speed is what the characteristic speed is driven
  /// toward in state Label::AhSpeed.
  double target_char_speed;
  /// target_drift_velocity is what dr/dt (where r and t are distorted frame
  /// variables) of the excision boundary is driven toward in state
  /// Label::Initial.
  double target_drift_velocity;
  /// Sometimes State::update will request that damping_time
  /// be changed; the new suggested value is suggested_time_scale. If it is a
  /// `std::nullopt` then there is no suggestion.
  std::optional<double> suggested_time_scale;
  /// discontinuous_change_has_occurred is set to true by
  /// State::update if it changes anything in such a way that
  /// the control signal jumps discontinuously in time.
  bool discontinuous_change_has_occurred;

  /// Reset `discontinuous_change_has_occurred` and `suggested_time_scale`
  void reset();

 private:
  void set_all_but_state(const Info& info);
};

/// Holds information about crossing times, as computed by
/// ZeroCrossingPredictors.
struct CrossingTimeInfo {
  CrossingTimeInfo(const double char_speed_crossing_time,
                   const double comoving_char_speed_crossing_time,
                   const double delta_radius_crossing_time);
  /// t_char_speed is the time (relative to the current time) when the
  /// minimum characteristic speed is predicted to cross zero (or zero if
  /// the minimum characteristic speed is increasing).
  double t_char_speed;
  /// t_comoving_char_speed is the time (relative to the current time) when the
  /// minimum comoving characteristic speed is predicted to cross zero
  /// (or zero if the minimum comoving characteristic speed is increasing).
  double t_comoving_char_speed;
  /// t_delta_radius is the time (relative to the current time) when the
  /// minimum distance between the horizon and the excision boundary is
  /// predicted to cross zero (or zero if the minimum distance is
  /// increasing).
  double t_delta_radius;
  /// Extra variables to simplify the logic; these indicate whether
  /// the characteristic speed or the excision boundary (or neither) are
  /// expected to cross zero soon.
  bool char_speed_will_hit_zero_first{false};
  bool horizon_will_hit_excision_boundary_first{false};
};
}  // namespace control_system::size
