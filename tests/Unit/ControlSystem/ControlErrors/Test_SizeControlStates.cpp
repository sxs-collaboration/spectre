// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <memory>
#include <optional>

#include "ControlSystem/ControlErrors/Size/AhSpeed.hpp"
#include "ControlSystem/ControlErrors/Size/DeltaR.hpp"
#include "ControlSystem/ControlErrors/Size/DeltaRDriftInward.hpp"
#include "ControlSystem/ControlErrors/Size/DeltaRDriftOutward.hpp"
#include "ControlSystem/ControlErrors/Size/DeltaRNoDrift.hpp"
#include "ControlSystem/ControlErrors/Size/Factory.hpp"
#include "ControlSystem/ControlErrors/Size/Info.hpp"
#include "ControlSystem/ControlErrors/Size/Initial.hpp"
#include "ControlSystem/ControlErrors/Size/RegisterDerivedWithCharm.hpp"
#include "ControlSystem/ControlErrors/Size/State.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"

namespace {

// Params passed into each test.
struct TestParams {
  // These are reasonable values for quantities that won't change in
  // the various logic tests.
  const double original_target_char_speed{0.011};
  const std::optional<double> average_radial_distance{
      0.01};  // This is what SpEC calls DeltaR.
  // The following means that the excision boundary radius in the grid frame
  // is 2.01.
  // Recall that horizon_00 is a Spherepack coefficient and not a raw
  // spherical harmonic coefficient.
  double horizon_00{4.02 * sqrt(2.0)};
  double avg_distorted_normal_dot_unit_coord_vector{1.0};
  // Defaults are values for quantities that we will vary so that the
  // logic makes different decisions.
  double damping_time{0.1};
  double min_char_speed{0.01};
  double min_comoving_char_speed{-0.02};
  double control_err_delta_r{0.03};
  std::optional<double> max_allowed_radial_distance{1.e100};
  // By default here we turn off state DeltaRDriftInward
  // by setting the following two options to nullopt.
  std::optional<double> min_allowed_radial_distance{std::nullopt};
  std::optional<double> min_allowed_char_speed{std::nullopt};
  std::optional<double> inward_drift_velocity{0.005};
  bool comoving_char_speed_increasing_inward{false};
  control_system::size::CrossingTimeInfo crossing_time_info{
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt};
};

template <typename InitialState, typename FinalState>
void do_test(const TestParams& test_params,
             const bool expected_discontinuous_change_has_occurred,
             const std::optional<double> expected_suggested_time_scale,
             const double expected_target_char_speed) {
  const std::string initial_state = InitialState{}.name();
  const std::string final_state = FinalState{}.name();
  CAPTURE(initial_state);
  CAPTURE(final_state);
  CAPTURE(expected_discontinuous_change_has_occurred);
  CAPTURE(expected_suggested_time_scale);
  CAPTURE(expected_target_char_speed);
  CAPTURE(test_params.original_target_char_speed);
  CAPTURE(test_params.damping_time);
  CAPTURE(test_params.min_char_speed);
  CAPTURE(test_params.min_comoving_char_speed);
  CAPTURE(test_params.control_err_delta_r);
  CAPTURE(test_params.max_allowed_radial_distance);
  CAPTURE(test_params.min_allowed_radial_distance);
  CAPTURE(test_params.min_allowed_char_speed);
  CAPTURE(test_params.inward_drift_velocity);
  CAPTURE(test_params.crossing_time_info.char_speed_will_hit_zero_first);
  CAPTURE(
      test_params.crossing_time_info.horizon_will_hit_excision_boundary_first);
  CAPTURE(test_params.crossing_time_info.t_char_speed);
  CAPTURE(test_params.crossing_time_info.t_comoving_char_speed);
  CAPTURE(test_params.crossing_time_info.t_delta_radius);
  CAPTURE(test_params.crossing_time_info.t_drift_limit_delta_radius);
  CAPTURE(test_params.crossing_time_info.t_drift_limit);
  CAPTURE(test_params.comoving_char_speed_increasing_inward);
  // Set reasonable values for quantities that won't change in the various
  // logic tests.
  const double target_drift_velocity = 0.001;
  const std::optional<double> original_suggested_time_scale = std::nullopt;
  const bool original_discontinuous_change_has_occurred = false;

  const control_system::size::StateUpdateArgs update_args{
      test_params.min_char_speed,
      test_params.min_comoving_char_speed,
      test_params.horizon_00,
      test_params.control_err_delta_r,
      test_params.average_radial_distance,
      test_params.max_allowed_radial_distance,
      test_params.avg_distorted_normal_dot_unit_coord_vector,
      test_params.inward_drift_velocity,
      test_params.min_allowed_radial_distance,
      test_params.min_allowed_char_speed,
      test_params.comoving_char_speed_increasing_inward};
  control_system::size::Info info{
      TestHelpers::test_factory_creation<control_system::size::State,
                                         InitialState>(initial_state),
      test_params.damping_time,
      test_params.original_target_char_speed,
      target_drift_velocity,
      original_suggested_time_scale,
      original_discontinuous_change_has_occurred};

  // Check serialization of info
  const auto info_copy = serialize_and_deserialize(info);
  CHECK_FALSE(info.state == nullptr);
  const auto info_copy2 = info_copy;
  CHECK_FALSE(info_copy2.state == nullptr);
  // Note that there is no equality operator for info.state, because the
  // state contains no data; so here we check that the state can be cast to
  // the type it should be.
  CHECK(dynamic_cast<InitialState*>(info_copy.state.get()) != nullptr);
  CHECK(info_copy.damping_time == info.damping_time);
  CHECK(info_copy.target_char_speed == info.target_char_speed);
  CHECK(info_copy.target_drift_velocity == info.target_drift_velocity);
  CHECK(info_copy.suggested_time_scale == info.suggested_time_scale);
  CHECK(info_copy.discontinuous_change_has_occurred ==
        info.discontinuous_change_has_occurred);

  auto state = info.state->get_clone();
  const std::string update_message = state->update(
      make_not_null(&info), update_args, test_params.crossing_time_info);
  CAPTURE(update_message);

  // These messages are hardcoded in the states
  CHECK(update_message.find("Current state " + initial_state) !=
        std::string::npos);
  CHECK(update_message.find_last_of(final_state) != std::string::npos);

  CHECK(info.state.get()->number() == FinalState{}.number());
  CHECK(info.damping_time == test_params.damping_time);
  CHECK(info.target_char_speed == expected_target_char_speed);
  CHECK(info.target_drift_velocity == target_drift_velocity);
  CHECK(info.suggested_time_scale == expected_suggested_time_scale);
  CHECK(info.discontinuous_change_has_occurred ==
        expected_discontinuous_change_has_occurred);

  info.reset();
  CHECK(info.damping_time == test_params.damping_time);
  CHECK(info.target_char_speed == expected_target_char_speed);
  CHECK(info.target_drift_velocity == target_drift_velocity);
  CHECK_FALSE(info.suggested_time_scale.has_value());
  CHECK_FALSE(info.discontinuous_change_has_occurred);
}

// For states X=Initial and X=AhSpeed, the logic for the transition
// from state X to state DeltaR is almost the same as the logic to
// transition from state X to state
// DeltaRDriftInward. test_transition_to_delta_r_inward encodes the
// differences between ending up in state DeltaR and ending up in
// state DeltaRDriftInward.  Every test below that ends up in state
// DeltaR from state Initial or from state AhSpeed should call
// test_transition_to_delta_r_inward afterwards.
template <typename InitialState>
void test_transition_to_delta_r_inward(
    TestParams test_params, const std::optional<double> suggested_time_scale,
    const double target_char_speed) {
  // Should_activate_inward_drift is true iff all of the following are true:
  //  1. inward_drift_velocity has a value.
  //  2. min_char_speed <= 0.9*min_allowed_char_speed or
  //     min_allowed_char_speed has no value
  //  3. avg_radial_distance <= 0.9*min_allowed_radial_distance or
  //     min_allowed_radial_distance has no value
  //  4. comoving_char_speed_increasing_inward is true
  //  5. min_allowed_char_speed has a value or
  //     min_allowed_radial_distance has a value
  //
  // should_transition_from_state_delta_r_to_inward_drift is true iff
  // all of the following are true:
  // A. should_activate_inward_drift is true
  // B. t_drift_limit >= damping time or t_drift_limit has no value
  //
  // should_transition_from_state_inward_drift_to_delta_r_no_drift is
  // true iff should_transition_from_state_delta_r_to_inward_drift is false.

  // On entry to this function, 1, 2, and 3 above are true, but 4 and 5
  // above are false.
  // Also, on entry to this function, t_drift_limit has no value so B.
  // is satisfied.

  // Here we make 4 true, but 5 is still false.
  // So 1,2,3,4 are true and 5 is false so we stay in state DeltaR.
  test_params.comoving_char_speed_increasing_inward = true;
  do_test<InitialState, control_system::size::States::DeltaR>(
      test_params, true, suggested_time_scale, target_char_speed);

  // Here we make 4 and 5 true, but 2 is now false (because limit is 0.9).
  // So 1,3,4,5 are true and 2 is false so we stay in state DeltaR.
  test_params.min_allowed_char_speed = test_params.min_char_speed / 0.91;
  do_test<InitialState, control_system::size::States::DeltaR>(
      test_params, true, suggested_time_scale, target_char_speed);

  // Here we make 4 and 5 true, but 2 is now false (because limit is 0.9)
  // and 3 is now false.
  // So 1,4,5 are true and 2,3 false so we stay in state DeltaR.
  test_params.min_allowed_radial_distance =
      test_params.average_radial_distance.value() / 0.91;
  do_test<InitialState, control_system::size::States::DeltaR>(
      test_params, true, suggested_time_scale, target_char_speed);

  // Here 1,2,4,5 are true and 3 false so we stay in state DeltaR.
  test_params.min_allowed_radial_distance =
      test_params.average_radial_distance.value() / 0.89;
  do_test<InitialState, control_system::size::States::DeltaR>(
      test_params, true, suggested_time_scale, target_char_speed);

  // Now all 1,2,3,4,5 are true.
  test_params.min_allowed_char_speed = test_params.min_char_speed / 0.89;
  do_test<InitialState, control_system::size::States::DeltaRDriftInward>(
      test_params, true, suggested_time_scale,
      std::min(test_params.inward_drift_velocity.value(),
               0.5 * test_params.min_char_speed /
                   test_params.avg_distorted_normal_dot_unit_coord_vector));

  // Now 4 above is false, even though 1,2,3,5 are true. So stay in State
  // DeltaR.
  test_params.comoving_char_speed_increasing_inward = false;
  do_test<InitialState, control_system::size::States::DeltaR>(
      test_params, true, suggested_time_scale, target_char_speed);
  test_params.comoving_char_speed_increasing_inward = true;

  // Now 1 above is false, even though 2,3,4,5 are true. So stay in State
  // DeltaR.
  test_params.inward_drift_velocity = std::nullopt;
  do_test<InitialState, control_system::size::States::DeltaR>(
      test_params, true, suggested_time_scale, target_char_speed);

  // 1,2,3,4,5 are true, so go to state DeltaRDriftInward.  This is
  // the same as the test above, but now the std::min in the last
  // argument takes a different value.
  test_params.inward_drift_velocity = 0.1;
  do_test<InitialState, control_system::size::States::DeltaRDriftInward>(
      test_params, true, suggested_time_scale,
      std::min(test_params.inward_drift_velocity.value(),
               0.5 * test_params.min_char_speed /
                   test_params.avg_distorted_normal_dot_unit_coord_vector));
}

void test_size_control_update() {
  TestParams test_params;  // With reasonable default values.

  // The parameters of the tests below are chosen by hand so that the
  // union of all the tests hit all of the 'if' statements in all of
  // the control_system::size::State::update functions.
  //
  // Each of the tests below is also done in SpEC (with the same input
  // parameters and the same expected results), to ensure that SpEC
  // and SpECTRE have the same size control logic.

  // First we do tests of state control_system::size::Label::Initial.

  // should do nothing
  do_test<control_system::size::States::Initial,
          control_system::size::States::Initial>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);

  // should go into DeltaR state
  test_params.min_comoving_char_speed = 0.02;
  do_test<control_system::size::States::Initial,
          control_system::size::States::DeltaR>(
      test_params, true, std::nullopt, test_params.original_target_char_speed);
  test_transition_to_delta_r_inward<control_system::size::States::Initial>(
      test_params, std::nullopt, test_params.original_target_char_speed);

  // Make deltar cross zero after damping time.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, 1.1 * test_params.damping_time, std::nullopt,
      std::nullopt);
  do_test<control_system::size::States::Initial,
          control_system::size::States::DeltaR>(
      test_params, true, std::nullopt, test_params.original_target_char_speed);
  test_transition_to_delta_r_inward<control_system::size::States::Initial>(
      test_params, std::nullopt, test_params.original_target_char_speed);

  // Make deltar cross zero before damping time.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, 0.9 * test_params.damping_time, std::nullopt,
      std::nullopt);
  do_test<control_system::size::States::Initial,
          control_system::size::States::DeltaR>(
      test_params, true, 0.9 * test_params.damping_time,
      test_params.original_target_char_speed);
  test_transition_to_delta_r_inward<control_system::size::States::Initial>(
      test_params, 0.9 * test_params.damping_time,
      test_params.original_target_char_speed);

  // Make deltar cross zero before damping time, faster than char speed.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      0.91 * test_params.damping_time, std::nullopt,
      0.899 * test_params.damping_time, std::nullopt, std::nullopt);
  do_test<control_system::size::States::Initial,
          control_system::size::States::DeltaR>(
      test_params, true, 0.899 * test_params.damping_time,
      test_params.original_target_char_speed);
  test_transition_to_delta_r_inward<control_system::size::States::Initial>(
      test_params, 0.899 * test_params.damping_time,
      test_params.original_target_char_speed);

  // Make deltar cross zero before damping time, same as char speed.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      0.9 * test_params.damping_time, std::nullopt,
      0.9 * test_params.damping_time, std::nullopt, std::nullopt);
  do_test<control_system::size::States::Initial,
          control_system::size::States::DeltaR>(
      test_params, true, 0.9 * test_params.damping_time,
      test_params.original_target_char_speed);
  test_transition_to_delta_r_inward<control_system::size::States::Initial>(
      test_params, 0.9 * test_params.damping_time,
      test_params.original_target_char_speed);

  // Make deltar cross zero before damping time, slower than char speed.
  // Now it goes to state AhSpeed.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      0.89 * test_params.damping_time, std::nullopt,
      0.9 * test_params.damping_time, std::nullopt, std::nullopt);
  do_test<control_system::size::States::Initial,
          control_system::size::States::AhSpeed>(
      test_params, true, 0.89 * test_params.damping_time,
      test_params.min_char_speed * 1.01);

  // Trigger DeltaRDriftOutward by changing max_allowed_radial_distance
  test_params.max_allowed_radial_distance = 0.001;
  // Make sure nothing is in danger.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
  // Comoving speed should be <0 or else we get state DeltaR and not
  // DeltaRDriftOutward.
  test_params.min_comoving_char_speed = -0.02;
  do_test<control_system::size::States::Initial,
          control_system::size::States::DeltaRDriftOutward>(
      test_params, true, std::nullopt, test_params.original_target_char_speed);
  // Turn off DeltaRDriftOutward again.
  test_params.max_allowed_radial_distance = std::nullopt;

  // Now do DeltaR tests
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt);

  // Should stay in DeltaR.
  do_test<control_system::size::States::DeltaR,
          control_system::size::States::DeltaR>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);

  // Should stay in DeltaR but change suggested timescale
  // because inward_drift_limit_in_danger.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, std::nullopt, std::nullopt,
      0.95 * test_params.damping_time);
  test_params.min_allowed_char_speed = test_params.min_char_speed / 0.98;
  test_params.min_allowed_radial_distance =
      test_params.average_radial_distance.value() / 0.98;
  do_test<control_system::size::States::DeltaR,
          control_system::size::States::DeltaR>(
      test_params, false, 0.95 * test_params.damping_time,
      test_params.original_target_char_speed);

  // Should stay in DeltaR because
  // t_drift_limit is less than damping time.
  test_params.comoving_char_speed_increasing_inward = true;
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, std::nullopt, 0.95 * test_params.damping_time,
      std::nullopt);
  test_params.min_allowed_char_speed = test_params.min_char_speed / 0.89;
  test_params.min_allowed_radial_distance =
      test_params.average_radial_distance.value() / 0.89;
  do_test<control_system::size::States::DeltaR,
          control_system::size::States::DeltaR>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);

  // Exactly the same but now all the crossing times are null,
  // so it should go to state DeltaRDriftInward with no change in timescale,
  // but a change in target_char_speed.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
  do_test<control_system::size::States::DeltaR,
          control_system::size::States::DeltaRDriftInward>(
      test_params, true, std::nullopt,
      std::min(test_params.inward_drift_velocity.value(),
               0.5 * test_params.min_char_speed /
                   test_params.avg_distorted_normal_dot_unit_coord_vector));

  // Should stay in state DeltaR if either CharSpeed or DeltaR
  // are above the min limits.
  // So test both cases and then put back the previous values of the limits.
  test_params.min_allowed_char_speed = test_params.min_char_speed / 0.91;
  do_test<control_system::size::States::DeltaR,
          control_system::size::States::DeltaR>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);
  test_params.min_allowed_char_speed = test_params.min_char_speed / 0.89;
  test_params.min_allowed_radial_distance =
      test_params.average_radial_distance.value() / 0.91;
  do_test<control_system::size::States::DeltaR,
          control_system::size::States::DeltaR>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);
  test_params.min_allowed_radial_distance =
      test_params.average_radial_distance.value() / 0.89;

  // Should stay in DeltaR if comoving_char_speed_increasing_inward is false.
  test_params.comoving_char_speed_increasing_inward = false;
  do_test<control_system::size::States::DeltaR,
          control_system::size::States::DeltaR>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);

  // Turn off DeltaRDriftInward again.
  test_params.min_allowed_char_speed = std::nullopt;
  test_params.min_allowed_radial_distance = std::nullopt;

  // Should change suggested time scale
  test_params.min_comoving_char_speed = 0.02;
  do_test<control_system::size::States::DeltaR,
          control_system::size::States::DeltaR>(
      test_params, false, 0.99 * test_params.damping_time,
      test_params.original_target_char_speed);

  // Should do nothing
  test_params.control_err_delta_r = 1.e-4;
  do_test<control_system::size::States::DeltaR,
          control_system::size::States::DeltaR>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);

  // Make deltar cross zero *slightly* before damping time; should do
  // nothing (depends on tolerance in control_system::size::StateDeltaR).
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, 0.999 * test_params.damping_time,
      std::nullopt, std::nullopt);
  do_test<control_system::size::States::DeltaR,
          control_system::size::States::DeltaR>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);

  // Make deltar cross zero before damping time.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, 0.9 * test_params.damping_time, std::nullopt,
      std::nullopt);
  do_test<control_system::size::States::DeltaR,
          control_system::size::States::DeltaR>(
      test_params, false, 0.9 * test_params.damping_time,
      test_params.original_target_char_speed);

  // Make deltar cross zero before damping time, faster than char speed.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      0.91 * test_params.damping_time, std::nullopt,
      0.9 * test_params.damping_time, std::nullopt, std::nullopt);
  do_test<control_system::size::States::DeltaR,
          control_system::size::States::DeltaR>(
      test_params, false, 0.9 * test_params.damping_time,
      test_params.original_target_char_speed);

  // Make deltar cross zero before damping time, same as char speed.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      0.9 * test_params.damping_time, std::nullopt,
      0.9 * test_params.damping_time, std::nullopt, std::nullopt);
  do_test<control_system::size::States::DeltaR,
          control_system::size::States::DeltaR>(
      test_params, false, 0.9 * test_params.damping_time,
      test_params.original_target_char_speed);

  // Make deltar cross zero before damping time, slower than char speed.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      0.89 * test_params.damping_time, std::nullopt,
      0.9 * test_params.damping_time, std::nullopt, std::nullopt);
  do_test<control_system::size::States::DeltaR,
          control_system::size::States::DeltaR>(
      test_params, false, 0.89 * test_params.damping_time,
      test_params.original_target_char_speed);

  // Same crossing_time_info, but comoving_char_speed is negative.
  // Should have different result as previous test.
  test_params.min_comoving_char_speed = -0.02;
  do_test<control_system::size::States::DeltaR,
          control_system::size::States::AhSpeed>(
      test_params, true, 0.89 * test_params.damping_time,
      test_params.min_char_speed * 1.01);

  // Same as 2 tests ago, but comoving_char_speed will cross zero far
  // in the future.  Should be same result as previous test.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      0.89 * test_params.damping_time, 1.e12, 0.9 * test_params.damping_time,
      std::nullopt, std::nullopt);
  test_params.min_comoving_char_speed = 0.02;
  do_test<control_system::size::States::DeltaR,
          control_system::size::States::AhSpeed>(
      test_params, true, 0.89 * test_params.damping_time,
      test_params.min_char_speed * 1.01);

  // Trigger DeltaRDriftOutward by changing max_allowed_radial_distance
  test_params.max_allowed_radial_distance = 0.001;
  // Make sure nothing is in danger.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
  // Comoving speed should be <0 or else we get state DeltaR and not
  // DeltaRDriftOutward.
  test_params.min_comoving_char_speed = -0.02;
  do_test<control_system::size::States::DeltaR,
          control_system::size::States::DeltaRDriftOutward>(
      test_params, true, std::nullopt, test_params.original_target_char_speed);

  // Turn off DeltaRDriftOutward
  test_params.max_allowed_radial_distance = std::nullopt;

  // Now do AhSpeed tests
  test_params.min_comoving_char_speed = -0.02;
  test_params.control_err_delta_r = 0.03;
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt);

  // Should do nothing.
  do_test<control_system::size::States::AhSpeed,
          control_system::size::States::AhSpeed>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);

  // Should change to DeltaR state.
  test_params.min_comoving_char_speed = 0.02;
  do_test<control_system::size::States::AhSpeed,
          control_system::size::States::DeltaR>(test_params, true, std::nullopt,
                                                0.0);
  test_transition_to_delta_r_inward<control_system::size::States::AhSpeed>(
      test_params, std::nullopt, 0.0);

  // Should do nothing because min_comoving_char_speed is smaller than
  // min_char_speed.
  test_params.min_comoving_char_speed = 0.99 * test_params.min_char_speed;
  do_test<control_system::size::States::AhSpeed,
          control_system::size::States::AhSpeed>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);

  // Now should change to DeltaR state if min_char_speed is larger than
  // target_char_speed
  test_params.min_char_speed = 0.012;
  do_test<control_system::size::States::AhSpeed,
          control_system::size::States::DeltaR>(test_params, true, std::nullopt,
                                                0.0);
  test_transition_to_delta_r_inward<control_system::size::States::AhSpeed>(
      test_params, std::nullopt, 0.0);

  // Now it should do nothing because comoving crossing time is very small.
  test_params.min_char_speed = 0.01;
  test_params.min_comoving_char_speed = 0.02;
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, 1.e-10, std::nullopt, std::nullopt, std::nullopt);
  do_test<control_system::size::States::AhSpeed,
          control_system::size::States::AhSpeed>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);

  // Now it should go to DeltaR because comoving crossing time is large.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, 100.0, std::nullopt, std::nullopt, std::nullopt);
  do_test<control_system::size::States::AhSpeed,
          control_system::size::States::DeltaR>(test_params, true, std::nullopt,
                                                0.0);
  test_transition_to_delta_r_inward<control_system::size::States::AhSpeed>(
      test_params, std::nullopt, 0.0);

  // Now it should do nothing because comoving is decreasing faster than
  // charspeeds.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      1000.0, 100.0, std::nullopt, std::nullopt, std::nullopt);
  do_test<control_system::size::States::AhSpeed,
          control_system::size::States::AhSpeed>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);

  // Now it should think delta_r is in danger,
  // and it should go to DeltaR state. And it should change the damping time.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, 19.0 * test_params.damping_time, std::nullopt,
      std::nullopt);
  do_test<control_system::size::States::AhSpeed,
          control_system::size::States::DeltaR>(
      test_params, true, 19.0 * test_params.damping_time, 0.0);
  test_transition_to_delta_r_inward<control_system::size::States::AhSpeed>(
      test_params, 19.0 * test_params.damping_time, 0.0);

  // But now with comoving_char_speed negative it should stay in AhSpeed
  // state, but with a change in target speed.
  test_params.min_comoving_char_speed = -0.02;
  do_test<control_system::size::States::AhSpeed,
          control_system::size::States::AhSpeed>(
      test_params, true, test_params.damping_time,
      0.125 * test_params.min_char_speed);

  // With min_comoving_char_speed positive, it should still stay in
  // AhSpeed state if char_speed has a positive crossing time.
  test_params.min_comoving_char_speed = 0.02;
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      1.e10, std::nullopt, 19.0 * test_params.damping_time, std::nullopt,
      std::nullopt);
  do_test<control_system::size::States::AhSpeed,
          control_system::size::States::AhSpeed>(
      test_params, true, test_params.damping_time,
      0.125 * test_params.min_char_speed);

  // .. but not if the delta_r crossing time is small enough.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      1.e10, std::nullopt, 4.99 * test_params.damping_time, std::nullopt,
      std::nullopt);
  do_test<control_system::size::States::AhSpeed,
          control_system::size::States::DeltaR>(
      test_params, true, 4.99 * test_params.damping_time, 0.0);
  test_transition_to_delta_r_inward<control_system::size::States::AhSpeed>(
      test_params, 4.99 * test_params.damping_time, 0.0);

  // If it thinks char speed is in danger, and the target char speed is
  // greater than the char speed, it changes the timescale and
  // nothing else.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      0.89 * test_params.damping_time, std::nullopt,
      0.9 * test_params.damping_time, std::nullopt, std::nullopt);
  do_test<control_system::size::States::AhSpeed,
          control_system::size::States::AhSpeed>(
      test_params, false, 0.89 * test_params.damping_time,
      test_params.original_target_char_speed);

  // ...but in the same situation, if char speed is greater than the
  // target speed, it resets the target speed too.
  test_params.min_char_speed = test_params.original_target_char_speed * 1.0001;
  do_test<control_system::size::States::AhSpeed,
          control_system::size::States::AhSpeed>(
      test_params, false, 0.89 * test_params.damping_time,
      test_params.min_char_speed * 1.01);

  // Same situation as previous, but char speed is *barely* in danger.
  test_params.min_char_speed = test_params.original_target_char_speed * 1.09999;
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      0.98999 * test_params.damping_time, std::nullopt,
      0.99 * test_params.damping_time, std::nullopt, std::nullopt);
  do_test<control_system::size::States::AhSpeed,
          control_system::size::States::AhSpeed>(
      test_params, false, 0.98999 * test_params.damping_time,
      test_params.min_char_speed * 1.01);

  // Same situation as previous, but char speed is *barely not* in danger,
  // and DeltaR is also not in danger.  Should go to DeltaR state.
  test_params.min_char_speed = test_params.original_target_char_speed * 1.10001;
  do_test<control_system::size::States::AhSpeed,
          control_system::size::States::DeltaR>(test_params, true, std::nullopt,
                                                0.0);
  test_transition_to_delta_r_inward<control_system::size::States::AhSpeed>(
      test_params, std::nullopt, 0.0);

  // Again char speed is *barely not* in danger, but for a different reason.
  test_params.min_char_speed = test_params.original_target_char_speed * 1.09999;
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      0.99001 * test_params.damping_time, std::nullopt,
      0.992 * test_params.damping_time, std::nullopt, std::nullopt);
  do_test<control_system::size::States::AhSpeed,
          control_system::size::States::DeltaR>(test_params, true, std::nullopt,
                                                0.0);
  test_transition_to_delta_r_inward<control_system::size::States::AhSpeed>(
      test_params, std::nullopt, 0.0);

  // Should go into state DeltaRDriftOutward.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
  test_params.min_comoving_char_speed = -0.02;
  test_params.min_char_speed = -0.01;
  test_params.max_allowed_radial_distance = 0.00069;
  do_test<control_system::size::States::AhSpeed,
          control_system::size::States::DeltaRDriftOutward>(
      test_params, true, std::nullopt, test_params.original_target_char_speed);

  // Should not go into state DeltaRDriftOutward because
  // CharSpeed is not in danger and its crossing time is valid.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      100.0, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
  do_test<control_system::size::States::AhSpeed,
          control_system::size::States::AhSpeed>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);

  // Now do DeltaRDriftOutward tests
  test_params.max_allowed_radial_distance = 0.001;
  test_params.min_char_speed = 0.01;
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt);

  // Should do nothing.
  do_test<control_system::size::States::DeltaRDriftOutward,
          control_system::size::States::DeltaRDriftOutward>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);

  // Make deltar cross zero *slightly* before damping time; it should still do
  // nothing (depends on tolerance in control_system::size::DeltaRDriftOutward).
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, 0.999 * test_params.damping_time,
      std::nullopt, std::nullopt);
  do_test<control_system::size::States::DeltaRDriftOutward,
          control_system::size::States::DeltaRDriftOutward>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);

  // Make charspeed cross zero slightly after damping time; it should
  // still do nothing.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      1.001 * test_params.damping_time, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt);
  do_test<control_system::size::States::DeltaRDriftOutward,
          control_system::size::States::DeltaRDriftOutward>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);

  // Make deltar cross zero before damping time. Now it should suggest
  // a new damping time.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, 0.9 * test_params.damping_time, std::nullopt,
      std::nullopt);
  do_test<control_system::size::States::DeltaRDriftOutward,
          control_system::size::States::DeltaRDriftOutward>(
      test_params, false, 0.9 * test_params.damping_time,
      test_params.original_target_char_speed);

  // Make deltar and charspeed cross zero before damping time, deltar
  // faster than char speed.  Should suggest new damping time.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      0.91 * test_params.damping_time, std::nullopt,
      0.9 * test_params.damping_time, std::nullopt, std::nullopt);
  do_test<control_system::size::States::DeltaRDriftOutward,
          control_system::size::States::DeltaRDriftOutward>(
      test_params, false, 0.9 * test_params.damping_time,
      test_params.original_target_char_speed);

  // Make deltar and charspeed cross zero before damping time, deltar
  // slower than char speed.  Should go to AhSpeed.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      0.89 * test_params.damping_time, std::nullopt,
      0.9 * test_params.damping_time, std::nullopt, std::nullopt);
  do_test<control_system::size::States::DeltaRDriftOutward,
          control_system::size::States::AhSpeed>(
      test_params, true, 0.89 * test_params.damping_time,
      1.01 * test_params.min_char_speed);

  // Should go to state DeltaR because distance < max_allowed_radial_distance.
  test_params.max_allowed_radial_distance = 1.e100;
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
  do_test<control_system::size::States::DeltaRDriftOutward,
          control_system::size::States::DeltaR>(
      test_params, true, std::nullopt, test_params.original_target_char_speed);

  // Tests for state DeltaRDriftInward.
  // If CharSpeed not in danger and DeltaR not in danger, then we
  // look at a few things.
  // First we check if we should transition to state DeltaRNoDrift.  To
  // transition to State DeltaRNoDrift, EITHER all of the following are true:
  //  1. t_drift_limit < tdamp
  //  2. t_drift_limit is valid
  //  3. inward_drift_velocity is nonzero
  // OR at least one of the following are true:
  //  4. inward_drift_velocity is nullopt
  //  5. min_char_speed > 0.9*min_allowed_char_speed and
  //     min_allowed_char_speed is valid
  //  6. avg_radial_distance > 0.9*min_allowed_radial_distance and
  //     min_allowed_radial_distance is valid
  //  7. comoving_char_speed_increasing_inward is false
  //  8. min_allowed_char_speed is invalid and
  //     min_allowed_radial_distance is invalid

  // Here t_drift_limit is invalid, so 1+2+3 is false.
  // But 8. above is true, so change to DeltaRNoDrift.
  test_params.comoving_char_speed_increasing_inward = true;
  test_params.min_allowed_radial_distance = std::nullopt;
  test_params.min_allowed_char_speed = std::nullopt;
  do_test<control_system::size::States::DeltaRDriftInward,
          control_system::size::States::DeltaRNoDrift>(
      test_params, true, std::nullopt, test_params.original_target_char_speed);

  // No transition because 8 above is no longer true.
  // (also 5 above is not true)
  test_params.min_allowed_char_speed = test_params.min_char_speed / 0.89;
  do_test<control_system::size::States::DeltaRDriftInward,
          control_system::size::States::DeltaRDriftInward>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);

  // No transition because 8 above is no longer true.
  // (also 6 above is not true)
  test_params.min_allowed_char_speed = std::nullopt;
  test_params.min_allowed_radial_distance =
      test_params.average_radial_distance.value() / 0.89;
  do_test<control_system::size::States::DeltaRDriftInward,
          control_system::size::States::DeltaRDriftInward>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);

  // Change to state DeltaRNoDrift with a timescale for t_drift_limit,
  // because t_drift_limit is now less than damping time.
  // Happens because 1,2,3 above are all true.
  // Note that 8 and 6 above are still false, i.e. all of 4 thru 8 are false.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, std::nullopt, std::nullopt,
      0.95 * test_params.damping_time);
  do_test<control_system::size::States::DeltaRDriftInward,
          control_system::size::States::DeltaRNoDrift>(
      test_params, true, test_params.crossing_time_info.t_drift_limit,
      test_params.original_target_char_speed);

  // Still change to State DeltaRNoDrift if CharSpeed is above the limit.
  // Note that 1. above is false.  Now 5 is true.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, std::nullopt, std::nullopt,
      1.2 * test_params.damping_time);
  test_params.min_allowed_char_speed = test_params.min_char_speed / 0.91;
  do_test<control_system::size::States::DeltaRDriftInward,
          control_system::size::States::DeltaRNoDrift>(
      test_params, true, test_params.crossing_time_info.t_drift_limit,
      test_params.original_target_char_speed);

  // Still change to State DeltaRNoDrift if DeltaR is above the limit.
  // Note that 1. above is false.  Now 5 and 6 are true.
  test_params.min_allowed_radial_distance =
      test_params.average_radial_distance.value() / 0.91;
  do_test<control_system::size::States::DeltaRDriftInward,
          control_system::size::States::DeltaRNoDrift>(
      test_params, true, test_params.crossing_time_info.t_drift_limit,
      test_params.original_target_char_speed);

  // Now put DeltaR below the limit. Still goes to state DeltaRNoDrift.
  // Note that 1. above is false.  Now 6 is true, but the rest of 4-8 are false.
  test_params.min_allowed_char_speed = test_params.min_char_speed / 0.89;
  do_test<control_system::size::States::DeltaRDriftInward,
          control_system::size::States::DeltaRNoDrift>(
      test_params, true, test_params.crossing_time_info.t_drift_limit,
      test_params.original_target_char_speed);

  // Now put DeltaR below the limit.
  // Now it doesn't transition because all of 4-8 are false,
  // and 1 is still false.
  test_params.min_allowed_radial_distance =
      test_params.average_radial_distance.value() / 0.89;
  do_test<control_system::size::States::DeltaRDriftInward,
          control_system::size::States::DeltaRDriftInward>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);

  // Goes to DeltaRNoDrift because 4 above is true.
  test_params.inward_drift_velocity = std::nullopt;
  do_test<control_system::size::States::DeltaRDriftInward,
          control_system::size::States::DeltaRNoDrift>(
      test_params, true, test_params.crossing_time_info.t_drift_limit,
      test_params.original_target_char_speed);

  // Goes to DeltaRNoDrift because 7 above is true.
  test_params.inward_drift_velocity = 0.005;
  test_params.comoving_char_speed_increasing_inward = false;
  do_test<control_system::size::States::DeltaRDriftInward,
          control_system::size::States::DeltaRNoDrift>(
      test_params, true, test_params.crossing_time_info.t_drift_limit,
      test_params.original_target_char_speed);
  test_params.comoving_char_speed_increasing_inward = true;

  // Now put DeltaR below the DeltaRDriftOutward limit.
  // Should go to DeltaRDriftOutward.
  test_params.max_allowed_radial_distance =
      test_params.average_radial_distance.value() / 3.5;
  do_test<control_system::size::States::DeltaRDriftInward,
          control_system::size::States::DeltaRDriftOutward>(
      test_params, true, std::nullopt, test_params.original_target_char_speed);
  test_params.max_allowed_radial_distance = std::nullopt;

  // Now both CharSpeed and DeltaR are below the limit, but
  // damping_time is large and t_drift_limit is active.  Now it should
  // stay in State DeltaRDriftInward but change the timescale.
  test_params.min_allowed_char_speed = test_params.min_char_speed / 0.89;
  test_params.damping_time = 5.0;
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, 20.0, std::nullopt,
      1.1 * test_params.damping_time);
  do_test<control_system::size::States::DeltaRDriftInward,
          control_system::size::States::DeltaRDriftInward>(
      test_params, false, 0.99 * test_params.damping_time,
      std::min(test_params.inward_drift_velocity.value(),
               0.5 * test_params.min_char_speed /
                   test_params.avg_distorted_normal_dot_unit_coord_vector));
  test_params.damping_time = 0.1;

  // Now do DeltaRInDanger. Should stay in State DeltaRDriftInward but with
  // different timescale.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, 0.9 * test_params.damping_time, std::nullopt,
      1.1 * test_params.damping_time);
  do_test<control_system::size::States::DeltaRDriftInward,
          control_system::size::States::DeltaRDriftInward>(
      test_params, false, test_params.crossing_time_info.t_delta_radius,
      test_params.original_target_char_speed);

  // Now do CharSpeedInDanger. Should go to State AhSpeed with new
  // damping time.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      0.89 * test_params.damping_time, std::nullopt,
      0.9 * test_params.damping_time, std::nullopt,
      1.1 * test_params.damping_time);
  do_test<control_system::size::States::DeltaRDriftInward,
          control_system::size::States::AhSpeed>(
      test_params, true, test_params.crossing_time_info.t_char_speed,
      test_params.min_char_speed * 1.01);

  // The following tests start in state DeltaRNoDrift

  // CharSpeed is in danger, but ComovingCharSpeed is positive, so
  // stays in State DeltaRNoDrift with new timescale.
  test_params.min_comoving_char_speed = 0.02;
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      0.89 * test_params.damping_time, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt);
  do_test<control_system::size::States::DeltaRNoDrift,
          control_system::size::States::DeltaRNoDrift>(
      test_params, false, test_params.crossing_time_info.t_char_speed,
      test_params.original_target_char_speed);

  // CharSpeed is in danger, but ComovingCharSpeed is negative, so
  // goes to State AhSpeed.
  test_params.min_comoving_char_speed = -0.02;
  do_test<control_system::size::States::DeltaRNoDrift,
          control_system::size::States::AhSpeed>(
      test_params, true, test_params.crossing_time_info.t_char_speed,
      test_params.min_char_speed * 1.01);

  // CharSpeed is in danger, ComovingCharSpeed is positive,
  // but ComovingCharSpeed is decreasing (i.e.
  // its crossing time is positive), so
  // goes to State AhSpeed.
  test_params.min_comoving_char_speed = 0.02;
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      0.89 * test_params.damping_time, 0.02, std::nullopt, std::nullopt,
      std::nullopt);
  do_test<control_system::size::States::DeltaRNoDrift,
          control_system::size::States::AhSpeed>(
      test_params, true, test_params.crossing_time_info.t_char_speed,
      test_params.min_char_speed * 1.01);

  // CharSpeed is in danger, ComovingCharSpeed negative,
  // ComovingCharSpeed is decreasing (i.e.
  // its crossing time is positive), so
  // goes to State AhSpeed.
  test_params.min_comoving_char_speed = -0.02;
  do_test<control_system::size::States::DeltaRNoDrift,
          control_system::size::States::AhSpeed>(
      test_params, true, test_params.crossing_time_info.t_char_speed,
      test_params.min_char_speed * 1.01);

  // DeltaRex is in danger, so stays in State DeltaRNoDrift with different
  // timescale.
  test_params.min_comoving_char_speed = 0.02;
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, 0.7 * test_params.damping_time, std::nullopt,
      std::nullopt);
  do_test<control_system::size::States::DeltaRNoDrift,
          control_system::size::States::DeltaRNoDrift>(
      test_params, false, test_params.crossing_time_info.t_delta_radius,
      test_params.original_target_char_speed);

  // To transition from DeltaRNoDrift to DeltaR, we require
  // at least one of the following to be true:
  // A. t_drift_limit = std::nullopt
  // B. delta_r > 0.99 min_allowed_radial_distance
  // C. char_speed > 0.99 min_allowed_char_speed
  //
  // If none of the above are true, then we stay in DeltaRNoDrift with
  // a different timescale if both of the following are true:
  // D. t_drift_limit < damping_time
  // E. Either min_allowed_radial_distance or min_allowed_char_speed
  //    has a value.
  //
  // If either D. or E. is false, then we stay in DeltaRNoDrift but
  // keep the same timescale.

  // A, B, and C above are false, and D. and E. are true.
  // We stay in State DeltaRNoDrift with a different timescale.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, std::nullopt, 0.3 * test_params.damping_time,
      std::nullopt);
  test_params.min_allowed_char_speed = test_params.min_char_speed / 0.98;
  test_params.min_allowed_radial_distance =
      test_params.average_radial_distance.value() / 0.98;
  do_test<control_system::size::States::DeltaRNoDrift,
          control_system::size::States::DeltaRNoDrift>(
      test_params, false, test_params.crossing_time_info.t_drift_limit,
      test_params.original_target_char_speed);

  // A, B are false, C. is true. (D. and E. are true but do not matter here).
  // We exit DeltaRNoDrift.
  test_params.min_allowed_char_speed = test_params.min_char_speed / 0.991;
  do_test<control_system::size::States::DeltaRNoDrift,
          control_system::size::States::DeltaR>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);

  // A, C are false, B. is true. (D. and E. are true but do not matter here).
  // We exit DeltaRNoDrift.
  test_params.min_allowed_char_speed = test_params.min_char_speed / 0.98;
  test_params.min_allowed_radial_distance =
      test_params.average_radial_distance.value() / 0.991;
  do_test<control_system::size::States::DeltaRNoDrift,
          control_system::size::States::DeltaR>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);

  // B, C are false, A. is true. (D. and E. are true but do not matter here).
  // We exit DeltaRNoDrift.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
  test_params.min_allowed_radial_distance =
      test_params.average_radial_distance.value() / 0.98;
  do_test<control_system::size::States::DeltaRNoDrift,
          control_system::size::States::DeltaR>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);

  // A, B are false, C true. (D. false and E. true).
  // We exit DeltaRNoDrift.
  test_params.min_allowed_char_speed = test_params.min_char_speed / 0.991;
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, std::nullopt, 1.1 * test_params.damping_time,
      std::nullopt);
  do_test<control_system::size::States::DeltaRNoDrift,
          control_system::size::States::DeltaR>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);

  // A, C are false, B true. (D. false and E. true).
  // We exit DeltaRNoDrift.
  test_params.min_allowed_char_speed = test_params.min_char_speed / 0.98;
  test_params.min_allowed_radial_distance =
      test_params.average_radial_distance.value() / 0.991;
  do_test<control_system::size::States::DeltaRNoDrift,
          control_system::size::States::DeltaR>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);

  // A, B, C are false. (D. false and E. true).
  // We stay in State DeltaRNoDrift but with the same timescale.
  test_params.min_allowed_radial_distance =
      test_params.average_radial_distance.value() / 0.98;
  do_test<control_system::size::States::DeltaRNoDrift,
          control_system::size::States::DeltaRNoDrift>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);

  // A, B, C are false. (D. true and E. false).
  // We stay in State DeltaRNoDrift but with the same timescale.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, std::nullopt, 0.77 * test_params.damping_time,
      std::nullopt);
  test_params.min_allowed_char_speed = std::nullopt;
  test_params.min_allowed_radial_distance = std::nullopt;
  do_test<control_system::size::States::DeltaRNoDrift,
          control_system::size::States::DeltaRNoDrift>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);

  // Here we want to enter state DeltaRDriftOutward.
  test_params.max_allowed_radial_distance =
      test_params.average_radial_distance.value() / 1.2;
  do_test<control_system::size::States::DeltaRNoDrift,
          control_system::size::States::DeltaRDriftOutward>(
      test_params, true, std::nullopt, test_params.original_target_char_speed);
}

void test_size_control_error() {
  // This is a very rudimentary test.  It just computes
  // the same thing as the thing it is testing, but coded differently.
  const control_system::size::ControlErrorArgs args{0.01, 0.03, 0.04, 1.2,
                                                    0.33};
  const control_system::size::Info info{
      std::make_unique<control_system::size::States::Initial>(),
      1.1,
      0.011,
      1.e-3,
      2.e-3,
      false};
  CHECK(control_system::size::States::Initial{}.control_error(info, args) ==
        -0.329);
  CHECK(control_system::size::States::AhSpeed{}.control_error(info, args) ==
        approx(0.001 * sqrt(4.0 * M_PI) / 1.2));
  CHECK(control_system::size::States::DeltaR{}.control_error(info, args) ==
        0.03);
  CHECK(control_system::size::States::DeltaRDriftOutward{}.control_error(
            info, args) == 0.04);
}

template <typename State>
void test_clone_and_serialization() {
  std::unique_ptr<control_system::size::State> state =
      std::make_unique<State>();

  // Note that we don't check equality here.  None of the derived
  // classes of control_system::size::State actually have data.
  // We just check that the types are correct.
  CHECK(dynamic_cast<State*>(serialize_and_deserialize(state).get()) !=
        nullptr);

  // Note that we don't check equality here.  None of the derived
  // classes of control_system::size::State actually have data.
  // We just check that the types are correct.
  CHECK(dynamic_cast<State*>(state->get_clone().get()) != nullptr);
}

void test_name_and_number() {
  const control_system::size::States::Initial initial{};
  const control_system::size::States::AhSpeed ah_speed{};
  const control_system::size::States::DeltaR delta_r{};
  const control_system::size::States::DeltaRDriftInward delta_r_drift_inward{};
  const control_system::size::States::DeltaRNoDrift delta_r_no_drift{};
  const control_system::size::States::DeltaRDriftOutward
      delta_r_drift_outward{};

  CHECK(initial.name() == "Initial"s);
  CHECK(initial.number() == 0_st);
  CHECK(ah_speed.name() == "AhSpeed"s);
  CHECK(ah_speed.number() == 1_st);
  CHECK(delta_r.name() == "DeltaR"s);
  CHECK(delta_r.number() == 2_st);
  CHECK(delta_r_drift_inward.name() == "DeltaRDriftInward"s);
  CHECK(delta_r_drift_inward.number() == 3_st);
  CHECK(delta_r_no_drift.name() == "DeltaRNoDrift"s);
  CHECK(delta_r_no_drift.number() == 4_st);
  CHECK(delta_r_drift_outward.name() == "DeltaRDriftOutward"s);
  CHECK(delta_r_drift_outward.number() == 5_st);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.ControlSystem.SizeControlStates", "[Domain][Unit]") {
  control_system::size::register_derived_with_charm();
  test_size_control_update();
  test_size_control_error();
  test_clone_and_serialization<control_system::size::States::Initial>();
  test_clone_and_serialization<control_system::size::States::AhSpeed>();
  test_clone_and_serialization<control_system::size::States::DeltaR>();
  test_clone_and_serialization<
      control_system::size::States::DeltaRDriftInward>();
  test_clone_and_serialization<control_system::size::States::DeltaRNoDrift>();
  test_clone_and_serialization<
      control_system::size::States::DeltaRDriftOutward>();
  test_name_and_number();
}
