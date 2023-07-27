// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>

#include "ControlSystem/Averager.hpp"
#include "ControlSystem/ControlErrors/Size.hpp"
#include "ControlSystem/ControlErrors/Size/AhSpeed.hpp"
#include "ControlSystem/ControlErrors/Size/DeltaR.hpp"
#include "ControlSystem/ControlErrors/Size/Error.hpp"
#include "ControlSystem/ControlErrors/Size/Info.hpp"
#include "ControlSystem/ControlErrors/Size/Initial.hpp"
#include "ControlSystem/ControlErrors/Size/RegisterDerivedWithCharm.hpp"
#include "ControlSystem/ControlErrors/Size/State.hpp"
#include "ControlSystem/ControlErrors/Size/Update.hpp"
#include "ControlSystem/Tags/QueueTags.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/ExcisionSphere.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Framework/TestCreation.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/Interpolation/ZeroCrossingPredictor.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Frame {
struct Distorted;
}

namespace {
struct Metavars {
  using const_global_cache_tags =
      tmpl::list<domain::Tags::FunctionsOfTimeInitialize,
                 domain::Tags::Domain<3>, control_system::Tags::WriteDataToDisk,
                 control_system::Tags::Verbosity>;
  using observed_reduction_data_tags = tmpl::list<>;
  using component_list = tmpl::list<observers::ObserverWriter<Metavars>>;
  void pup(PUP::er& /*p*/) {}
};

void test_control_error_delta_r() {
  const double horizon_00 = 2.0;
  const double dt_horizon_00 = 1.0;
  const double lambda_00 = 3.0;
  const double dt_lambda_00 = 4.0;
  // This is 0 so we avoid the term with Y00 so we can get an (easy) exact
  // calculation
  const double grid_frame_excision_radius = 0.0;

  const double control_error_delta_r =
      control_system::size::control_error_delta_r(horizon_00, dt_horizon_00,
                                                  lambda_00, dt_lambda_00,
                                                  grid_frame_excision_radius);

  CHECK(control_error_delta_r == approx(-2.5));
}

template <typename InitialState, typename FinalState>
void test_size_error_one_step(
    const gsl::not_null<intrp::ZeroCrossingPredictor*> predictor_char_speed,
    const gsl::not_null<intrp::ZeroCrossingPredictor*>
        predictor_comoving_char_speed,
    const gsl::not_null<intrp::ZeroCrossingPredictor*> predictor_delta_radius,
    const double time, const double grid_excision_boundary_radius,
    const double distorted_excision_boundary_radius_initial,
    const double distorted_excision_boundary_velocity,
    const double distorted_horizon_velocity, const double target_char_speed,
    const std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>&
        function_of_time,
    const double expected_error) {
  const std::string initial_state = pretty_type::name<InitialState>();
  const std::string final_state = pretty_type::name<FinalState>();
  CAPTURE(initial_state);
  CAPTURE(final_state);
  const double initial_damping_time = 0.1;
  const double initial_target_drift_velocity = 0.0;
  const double initial_suggested_time_scale = 0.0;
  control_system::size::Info info{std::make_unique<InitialState>(),
                                  initial_damping_time,
                                  target_char_speed,
                                  initial_target_drift_velocity,
                                  initial_suggested_time_scale,
                                  false};

  const size_t l_max = 8;
  const double distorted_horizon_radius = 2.00;

  const std::array<double, 3> center{{0.0, 0.0, 0.0}};
  Strahlkorper<Frame::Distorted> horizon(l_max, distorted_horizon_radius,
                                         center);
  Strahlkorper<Frame::Distorted> excision_boundary(
      l_max, distorted_excision_boundary_radius_initial, center);
  Strahlkorper<Frame::Distorted> time_deriv_horizon(
      l_max, distorted_horizon_velocity, center);

  // Get Cartesian coordinates on excision boundary
  // (and other temp variables that are necessary to compute it).
  Variables<
      tmpl::list<::Tags::Tempi<0, 2, ::Frame::Spherical<Frame::Distorted>>,
                 ::Tags::Tempi<1, 3, Frame::Distorted>,
                 ::Tags::TempI<2, 3, Frame::Distorted>,
                 ::Tags::TempI<3, 3, Frame::Distorted>, ::Tags::TempScalar<4>>>
      temp_buffer(excision_boundary.ylm_spherepack().physical_size());
  auto& theta_phi =
      get<::Tags::Tempi<0, 2, ::Frame::Spherical<Frame::Distorted>>>(
          temp_buffer);
  auto& r_hat = get<::Tags::Tempi<1, 3, Frame::Distorted>>(temp_buffer);
  auto& cartesian_coords =
      get<Tags::TempI<2, 3, Frame::Distorted>>(temp_buffer);
  auto& shifty_quantity = get<Tags::TempI<3, 3, Frame::Distorted>>(temp_buffer);
  auto& radius = get<::Tags::TempScalar<4>>(temp_buffer);
  StrahlkorperTags::ThetaPhiCompute<Frame::Distorted>::function(
      make_not_null(&theta_phi), excision_boundary);
  StrahlkorperTags::RhatCompute<Frame::Distorted>::function(
      make_not_null(&r_hat), theta_phi);
  StrahlkorperTags::RadiusCompute<Frame::Distorted>::function(
      make_not_null(&radius), excision_boundary);
  StrahlkorperTags::CartesianCoordsCompute<Frame::Distorted>::function(
      make_not_null(&cartesian_coords), excision_boundary, radius, r_hat);

  // Get analytic Schwarzschild solution
  gr::Solutions::KerrSchild solution(
      1.0, std::array<double, 3>{{0.0, 0.0, 0.0}}, center);
  const auto vars = solution.variables(
      cartesian_coords, time,
      typename gr::Solutions::KerrSchild::tags<DataVector, Frame::Distorted>{});
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(vars);
  const auto& shift =
      get<gr::Tags::Shift<DataVector, 3, Frame::Distorted>>(vars);
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, 3, Frame::Distorted>>(vars);
  const auto& inverse_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<DataVector, 3, Frame::Distorted>>(
          vars);

  // Now compute shifty quantity, which is distorted shift plus
  // grid-to-distorted frame-velocity.
  for (size_t i = 0; i < 3; ++i) {
    shifty_quantity.get(i) =
        shift.get(i) + distorted_excision_boundary_velocity *
                           cartesian_coords.get(i) /
                           distorted_excision_boundary_radius_initial;
  }
  auto error = control_system::size::control_error(
      make_not_null(&info), predictor_char_speed, predictor_comoving_char_speed,
      predictor_delta_radius, time, horizon, excision_boundary,
      grid_excision_boundary_radius, time_deriv_horizon, lapse, shifty_quantity,
      spatial_metric, inverse_spatial_metric, function_of_time);

  // Check error and parts of info.
  //
  // Note that Test_SizeControlStates does extensive tests
  // of control_system::size::State::update() and
  // control_system::size::State::control_error(), which are the
  // main thing that happens inside of control_error.
  // Here we merely check that control_error does the correct
  // thing for a few cases.
  CHECK(dynamic_cast<FinalState*>(info.state.get()) != nullptr);
  CHECK(error.control_error == approx(expected_error));

  // Now check the control error class, but only if the initial state is Initial
  // because the control error class is hard coded to start in the Initial state
  if constexpr (std::is_same_v<InitialState,
                               control_system::size::States::Initial>) {
    using size_error =
        control_system::ControlErrors::Size<2, domain::ObjectLabel::A>;
    static_assert(
        tt::assert_conforms_to_v<size_error,
                                 control_system::protocols::ControlError>);

    auto error_class = TestHelpers::test_creation<size_error>(
        "MaxNumTimesForZeroCrossingPredictor: 4");

    CHECK_FALSE(error_class.get_suggested_timescale().has_value());
    CHECK_FALSE(error_class.discontinuous_change_has_occurred());

    TimescaleTuner tuner{
        std::vector<double>{0.1}, 1.0, 0.01, 1.0e-3, 1.0e-4, 1.01, 0.98};
    std::unordered_map<std::string,
                       std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
        functions_of_time{};
    functions_of_time["Size"] = function_of_time->get_clone();
    Domain<3> domain{
        {},
        {{"ExcisionSphereA", ExcisionSphere<3>{grid_excision_boundary_radius,
                                               tnsr::I<double, 3, Frame::Grid>{
                                                   std::array{0.0, 0.0, 0.0}},
                                               {}}}}};
    Parallel::GlobalCache<Metavars> cache{
        {std::move(functions_of_time), std::move(domain), false,
         ::Verbosity::Silent, "", "", std::vector<std::string>{}}};
    using ExcisionQuantities =
        control_system::QueueTags::SizeExcisionQuantities<Frame::Distorted>;
    using HorizonQuantities =
        control_system::QueueTags::SizeHorizonQuantities<Frame::Distorted>;
    tuples::TaggedTuple<ExcisionQuantities, HorizonQuantities> measurements{
        ExcisionQuantities::type{excision_boundary, lapse, shifty_quantity,
                                 spatial_metric, inverse_spatial_metric},
        HorizonQuantities::type{horizon, time_deriv_horizon}};

    const double control_error_from_class =
        error_class(tuner, cache, time, "Size"s, measurements)[0];
    const auto control_error_history = error_class.control_error_history();

    // These should be identical because the control error class calls the
    // control_error function
    CHECK(control_error_from_class == error.control_error);

    CHECK_FALSE(error_class.get_suggested_timescale().has_value());
    CHECK(error_class.discontinuous_change_has_occurred() !=
          std::is_same_v<InitialState, FinalState>);
    // The current time is popped back so we only get times in the past
    CHECK(control_error_history.empty());

    size_error error_class_copied = error_class;

    // We test the update_averager and update_tuner functions here because we
    // already have the nice infrastructure of a cache and the control error
    // class.
    Averager<1> averager{0.25, true};
    const DataVector timescale{1, 0.1};
    // Populate the averager so it will have data
    averager.update(time - 1.0, DataVector{1, 0.0}, timescale);
    averager.update(time, DataVector{1, 0.0}, timescale);
    CHECK(averager(0.0).has_value());

    const bool expected_discontinuous_change =
        error_class.discontinuous_change_has_occurred();
    // If a discontinuous change has occurred, this call will clear the control
    // error and the averager, then repopulate the averager with the existing
    // control error history. However, since there is no history (checked
    // above), the operator() of the averager will return a nullopt. If there
    // wasn't a discontinuous change, then nothing happens
    control_system::size::update_averager(make_not_null(&averager),
                                          make_not_null(&error_class), cache,
                                          time, timescale, "Size"s, 2);

    CHECK(averager(0.0).has_value() != expected_discontinuous_change);

    error_class = error_class_copied;

    const DataVector old_timescale = tuner.current_timescale();

    control_system::size::update_tuner(make_not_null(&tuner),
                                       make_not_null(&error_class), cache, time,
                                       "Size"s);

    // Since there is no suggested timescale, the tuner keeps its old timescale.
    // However the control error is always reset.
    CHECK(old_timescale == tuner.current_timescale());

    CHECK_FALSE(error_class.get_suggested_timescale().has_value());
    CHECK_FALSE(error_class.discontinuous_change_has_occurred());
    CHECK(control_error_history.empty());
  }
}

template <typename InitialState, typename FinalState>
void test_size_error(const double grid_excision_boundary_radius,
                     const double distorted_excision_boundary_radius_initial,
                     const double distorted_excision_boundary_velocity,
                     const double distorted_horizon_velocity,
                     const double target_char_speed,
                     const double expected_error) {
  const double initial_time = 0.0;
  const double Y00 = 0.25 * M_2_SQRTPI;
  std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime> function_of_time(
      new domain::FunctionsOfTime::PiecewisePolynomial<3>(
          initial_time,
          std::array<DataVector, 4>{
              {{(grid_excision_boundary_radius -
                 distorted_excision_boundary_radius_initial) /
                Y00},
               {-distorted_excision_boundary_velocity / Y00},
               {0.0},
               {0.0}}},
          std::numeric_limits<double>::infinity()));

  intrp::ZeroCrossingPredictor predictor_char_speed;
  intrp::ZeroCrossingPredictor predictor_comoving_char_speed;
  intrp::ZeroCrossingPredictor predictor_delta_radius;

  test_size_error_one_step<InitialState, FinalState>(
      make_not_null(&predictor_char_speed),
      make_not_null(&predictor_comoving_char_speed),
      make_not_null(&predictor_delta_radius), initial_time,
      grid_excision_boundary_radius, distorted_excision_boundary_radius_initial,
      distorted_excision_boundary_velocity, distorted_horizon_velocity,
      target_char_speed, function_of_time, expected_error);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ControlSystem.SizeError", "[Domain][Unit]") {
  control_system::size::register_derived_with_charm();
  test_control_error_delta_r();
  // Should go to DeltaR state with error of zero, since ComovingMinCharSpeed
  // will be positive.
  test_size_error<control_system::size::States::Initial,
                  control_system::size::States::DeltaR>(1.98, 1.98, 0.0, 0.0,
                                                        0.0, 0.0);
  // Should remain in Initial state, since ComovingMinCharSpeed will
  // be negative.  Note that the way we make ComovingMinCharSpeed negative
  // is we put the excision boundary outside (!) the horizon, which normally
  // should never happen but here it serves the purpose of this test.
  test_size_error<control_system::size::States::Initial,
                  control_system::size::States::Initial>(2.01, 2.01, 0.0, 0.0,
                                                         0.0, 0.0);
  const double Y00 = 0.25 * M_2_SQRTPI;
  const double horizon_velocity = 0.01;
  const double excision_velocity = 0.03;
  const double excision_grid = 1.95;
  const double target_char_speed = 0.05;
  {
    // The following is computed by hand from arxiv:1211.6079 eq. 96.
    const double excision_distorted = 1.98;
    const double expected_control_error =
        (-horizon_velocity * 0.5 * excision_distorted + excision_velocity) /
        Y00;
    // Should stay in state DeltaR.
    test_size_error<control_system::size::States::DeltaR,
                    control_system::size::States::DeltaR>(
        excision_grid, excision_distorted, excision_velocity, horizon_velocity,
        target_char_speed, expected_control_error);
  }

  {
    // The following is computed by hand from arxiv:1211.6079 eq. 92
    // and the Schwarzshild solution in Kerr-schild coords.
    const double excision_distorted = 2.0;
    const double mass = 1.0;  // hardcoded in test.
    const double normal_radial_one_form_mag =
        sqrt(1.0 + 2.0 * mass / excision_distorted);
    const double lapse = 1.0 / sqrt(1.0 + 2.0 * mass / excision_distorted);
    const double radial_shift_mag = 2.0 * mass / excision_distorted /
                                    (1.0 + 2.0 * mass / excision_distorted);
    const double shifty_quantity_mag = radial_shift_mag + excision_velocity;
    const double char_speed =
        -lapse + normal_radial_one_form_mag * shifty_quantity_mag;
    const double expected_control_error =
        (char_speed - target_char_speed) / (Y00 * normal_radial_one_form_mag);
    // Should stay in state AhSpeed.
    test_size_error<control_system::size::States::AhSpeed,
                    control_system::size::States::AhSpeed>(
        excision_grid, excision_distorted, excision_velocity, horizon_velocity,
        target_char_speed, expected_control_error);
  }
}
