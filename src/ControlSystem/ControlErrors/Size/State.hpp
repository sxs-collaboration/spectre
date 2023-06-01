// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <pup.h>

#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"

/// \cond
namespace control_system::size {
struct Info;
}  // namespace control_system::size
/// \endcond

namespace control_system::size {

/// Packages some of the inputs to the State::update, so
/// that State::update doesn't need a large number of
/// arguments.
struct StateUpdateArgs {
  /// min_char_speed is the minimum over the excision boundary
  /// of Eq. 89 of \cite Hemberger2012jz.
  double min_char_speed;
  /// min_comoving_char_speed is the minimum over the excision boundary
  /// of Eq. 28 of \cite Hemberger2012jz.
  double min_comoving_char_speed;
  /// control_error_delta_r is the control error when the control system
  /// is in state Label::DeltaR.
  /// This is Q in Eq. 96 of \cite Hemberger2012jz.
  double control_error_delta_r;
};

/// Packages some of the inputs to the State::control_error, so
/// that State::control_error doesn't need a large number of
/// arguments.
struct ControlErrorArgs {
  double min_char_speed;
  double control_error_delta_r;
  /// avg_distorted_normal_dot_unit_coord_vector is the average of
  /// distorted_normal_dot_unit_coord_vector over the excision
  /// boundary.  Here distorted_normal_dot_unit_coord_vector is Eq. 93
  /// of \cite Hemberger2012jz.  distorted_normal_dot_unit_coord_vector is
  /// \f$\hat{n}_i x^i/r\f$ where \f$\nat{n}_i\f$ is the
  /// distorted-frame unit normal to the excision boundary (pointing
  /// INTO the hole, i.e. out of the domain), and \f$x^i/r\f$ is the
  /// distorted-frame (or equivalently the grid frame because it is
  /// invariant between these two frames because of the required
  /// limiting behavior of the map we choose) Euclidean normal vector
  /// from the center of the excision-boundary Strahlkorper to each
  /// point on the excision-boundary Strahlkorper.
  double avg_distorted_normal_dot_unit_coord_vector;
  /// time_deriv_of_lambda_00 is the time derivative of the quantity lambda_00
  /// that appears in \cite Hemberger2012jz.  time_deriv_of_lambda_00 is (minus)
  /// the radial velocity of the excision boundary in the distorted frame with
  /// respect to the grid frame.
  double time_deriv_of_lambda_00;
};

/// Represents a 'state' of the size control system.
///
/// Each 'state' of the size control system has a different control
/// signal, which has a different purpose, even though each state
/// controls the same map quantity, namely the Y00 coefficient of the
/// shape map.  For example, state Label::AhSpeed controls
/// the Y00 coefficient of the shape map so that the minimum
/// characteristic speed is driven towards a target value, and state
/// Label::DeltaR controls the Y00 coefficient of the shape
/// map (or the Y00 coefficient of a separate spherically-symmetric size
/// map) so that the minimum difference between the horizon radius and
/// the excision boundary radius is driven towards a constant.
///
/// Each state has its own logic (the 'update' function) that
/// determines values of certain parameters (i.e. the things in
/// Info), including whether the control system should
/// transition to a different state.
///
/// The different states are:
/// - Initial: drives dr/dt of the excision boundary to
///   Info::target_drift_velocity.
/// - AhSpeed: drives the minimum characteristic speed on the excision boundary
///   to Info::target_char_speed.
/// - DeltaR: drives the minimum distance between the horizon and the excision
///   boundary to be constant in time.
/// - DeltaRDriftInward: Same as DeltaR but the excision boundary has a small
///   velocity inward.  This state is triggered when it is deemed that the
///   excision boundary and the horizon are too close to each other; the
///   small velocity makes the excision boundary and the horizon drift apart.
/// - DeltaRDriftOutward: Same as DeltaR but the excision boundary has a small
///   velocity outward.  This state is triggered when it is deemed that the
///   excision boundary and the horizon are too far apart.
/// - DeltaRTransition: Same as DeltaR except for the logic that
///   determines how DeltaRTransition changes to other states.
///   DeltaRTransition is allowed (under some circumstances) to change
///   to state DeltaR, but DeltaRDriftOutward and DeltaRDriftInward
///   are never allowed to change to state DeltaR.  Instead
///   DeltaRDriftOutward and DeltaRDriftInward are allowed (under
///   some circumstances) to change to state DeltaRTransition.
///
/// The reason that DeltaRDriftInward, DeltaRDriftOutward, and
/// DeltaRTransition are separate states is to simplify the logic.  In
/// principle, all 3 of those states could be merged with state
/// DeltaR, because the control error is the same for all four states
/// (except for a velocity term that could be set to zero).  But if that
/// were done, then there would need to be additional complicated
/// logic in determining transitions between different states, and
/// that logic would depend not only on the current state, but also on
/// the previous state.
class State : public PUP::able {
 public:
  State() = default;
  State(const State& /*rhs*/) = default;
  State& operator=(const State& /*rhs*/) = default;
  State(State&& /*rhs*/) = default;
  State& operator=(State&& /*rhs*/) = default;
  virtual ~State() override = default;

  /// Name of this state
  virtual std::string name() const = 0;

  /// Return a size_t that corresponds to the state number in SpEC
  virtual size_t number() const = 0;

  virtual std::unique_ptr<State> get_clone() const = 0;
  /// Updates the Info in `info`.  Notice that `info`
  /// includes a state, which might be different than the current
  /// state upon return. It is the caller's responsibility to check
  /// if the current state has changed.
  virtual void update(const gsl::not_null<Info*> info,
                      const StateUpdateArgs& update_args,
                      const CrossingTimeInfo& crossing_time_info) const = 0;
  /// Returns the control signal, but does not modify the state or any
  /// parameters.
  virtual double control_error(
      const Info& info, const ControlErrorArgs& control_error_args) const = 0;

  WRAPPED_PUPable_abstract(State);  // NOLINT
  explicit State(CkMigrateMessage* msg) : PUP::able(msg) {}
};
}  // namespace control_system::size
