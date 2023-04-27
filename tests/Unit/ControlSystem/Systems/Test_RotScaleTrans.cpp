// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "ControlSystem/Tags/MeasurementTimescales.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/TimeDependent/CubicScale.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Rotation.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/ControlSystem/SystemHelpers.hpp"
#include "Helpers/PointwiseFunctions/PostNewtonian/BinaryTrajectories.hpp"
#include "Parallel/Phase.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace Frame {
struct Grid;
struct Distorted;
struct Inertial;
}  // namespace Frame

namespace control_system {
namespace {
using TranslationMap = domain::CoordinateMaps::TimeDependent::Translation<3>;
using RotationMap = domain::CoordinateMaps::TimeDependent::Rotation<3>;
using ExpansionMap = domain::CoordinateMaps::TimeDependent::CubicScale<3>;

using CoordMap =
    domain::CoordinateMap<Frame::Distorted, Frame::Inertial, ExpansionMap,
                          RotationMap, TranslationMap>;

std::string create_input_string(const std::string& name) {
  const std::string name_str = "  "s + name + ":\n"s;
  const std::string base_string1{
      "    IsActive: true\n"
      "    Averager:\n"
      "      AverageTimescaleFraction: 0.25\n"
      "      Average0thDeriv: true\n"
      "    Controller:\n"
      "      UpdateFraction: 0.3\n"
      "    TimescaleTuner:\n"
      "      InitialTimescales: 0.5\n"
      "      MinTimescale: 0.1\n"
      "      MaxTimescale: 10.\n"
      "      DecreaseThreshold: 2.0\n"
      "      IncreaseThreshold: 0.1\n"
      "      IncreaseFactor: 1.01\n"
      "      DecreaseFactor: 0.99\n"
      "    ControlError:\n"};

  return name_str + base_string1;
}

template <size_t TranslationDerivOrder, size_t RotationDerivOrder,
          size_t ExpansionDerivOrder>
void test_rotscaletrans_control_system(const double rotation_eps = 5.0e-5) {
  INFO("Translation: "s + get_output(TranslationDerivOrder) + ", Rotation: "s +
       get_output(RotationDerivOrder) + ", Expansion: "s +
       get_output(ExpansionDerivOrder));
  using metavars =
      TestHelpers::MockMetavars<TranslationDerivOrder, RotationDerivOrder,
                                ExpansionDerivOrder, 0>;
  using element_component = typename metavars::element_component;
  using translation_component = typename metavars::translation_component;
  using rotation_component = typename metavars::rotation_component;
  using expansion_component = typename metavars::expansion_component;
  using translation_system = typename metavars::translation_system;
  using rotation_system = typename metavars::rotation_system;
  using expansion_system = typename metavars::expansion_system;
  MAKE_GENERATOR(gen);

  // Global things
  domain::FunctionsOfTime::register_derived_with_charm();
  const double initial_time = 0.0;
  const double initial_separation = 15.0;
  // This final time is chosen so that the damping timescales have adequate time
  // to reach the maximum damping timescale
  const double final_time = 500.0;

  // Set up the system helper
  control_system::TestHelpers::SystemHelper<metavars> system_helper{};

  const std::string translation_name =
      system_helper.template name<translation_system>();
  const std::string rotation_name =
      system_helper.template name<rotation_system>();
  const std::string expansion_name =
      system_helper.template name<expansion_system>();

  std::string input_options =
      "Evolution:\n"
      "  InitialTime: 0.0\n"
      "DomainCreator:\n"
      "  FakeCreator:\n"
      "    NumberOfExcisions: 2\n"
      "    NumberOfComponents:\n"
      "      Translation: 3\n"
      "      Rotation: 3\n"
      "      Expansion: 1\n"
      "ControlSystems:\n"
      "  WriteDataToDisk: false\n"
      "  MeasurementsPerUpdate: 4\n";
  input_options += create_input_string(translation_name);
  input_options += create_input_string(rotation_name);
  input_options += create_input_string(expansion_name);

  const auto initialize_functions_of_time =
      [](const gsl::not_null<std::unordered_map<
             std::string,
             std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>*>
             functions_of_time,
         const double local_initial_time,
         const std::unordered_map<std::string, double>&
             initial_expiration_times) {
        TestHelpers::initialize_expansion_functions_of_time<expansion_system>(
            functions_of_time, local_initial_time, initial_expiration_times);
        TestHelpers::initialize_rotation_functions_of_time<rotation_system>(
            functions_of_time, local_initial_time, initial_expiration_times);
        return TestHelpers::initialize_translation_functions_of_time<
            translation_system>(functions_of_time, local_initial_time,
                                initial_expiration_times);
      };

  // Initialize everything within the system helper
  system_helper.setup_control_system_test(initial_time, initial_separation,
                                          input_options,
                                          initialize_functions_of_time);

  // Get references to everything that was set up inside the system helper. The
  // domain and two functions of time are not const references because they need
  // to be moved into the runner
  auto& domain = system_helper.domain();
  auto& initial_functions_of_time = system_helper.initial_functions_of_time();
  auto& initial_measurement_timescales =
      system_helper.initial_measurement_timescales();
  const auto& init_trans_tuple =
      system_helper.template init_tuple<translation_system>();
  const auto& init_rot_tuple =
      system_helper.template init_tuple<rotation_system>();
  const auto& init_exp_tuple =
      system_helper.template init_tuple<expansion_system>();

  // Setup runner and all components
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  MockRuntimeSystem runner{
      {"DummyFileName", std::move(domain), 4, false, ::Verbosity::Silent,
       tnsr::I<double, 3, Frame::Grid>{{0.5 * initial_separation, 0.0, 0.0}},
       tnsr::I<double, 3, Frame::Grid>{{-0.5 * initial_separation, 0.0, 0.0}}},
      {std::move(initial_functions_of_time),
       std::move(initial_measurement_timescales)}};
  ActionTesting::emplace_singleton_component_and_initialize<
      translation_component>(make_not_null(&runner), ActionTesting::NodeId{0},
                             ActionTesting::LocalCoreId{0}, init_trans_tuple);
  ActionTesting::emplace_singleton_component_and_initialize<rotation_component>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0}, init_rot_tuple);
  ActionTesting::emplace_singleton_component_and_initialize<
      expansion_component>(make_not_null(&runner), ActionTesting::NodeId{0},
                           ActionTesting::LocalCoreId{0}, init_exp_tuple);
  ActionTesting::emplace_array_component<element_component>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0}, 0);

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  // PN orbits
  const std::array<double, 3> initial_velocity{0.1, -0.2, 0.3};
  const BinaryTrajectories binary_trajectories{initial_separation,
                                               initial_velocity};

  // Create coordinate maps for mapping the PN trajectories to the "grid" frame
  // where the control system does its calculations
  TranslationMap translation_map{translation_name};
  RotationMap rotation_map{rotation_name};
  // The outer boundary is at 1000.0 so that we don't have to worry about it.
  ExpansionMap expansion_map{1000.0, expansion_name,
                             expansion_name + "OuterBoundary"s};

  CoordMap coord_map{expansion_map, rotation_map, translation_map};

  // Get the functions of time from the cache to use in the maps
  const auto& cache = ActionTesting::cache<element_component>(runner, 0);
  const auto& functions_of_time =
      Parallel::get<domain::Tags::FunctionsOfTime>(cache);

  const auto position_function = [&binary_trajectories](const double time) {
    return binary_trajectories.positions(time);
  };

  const auto horizon_function = [&position_function, &runner,
                                 &coord_map](const double time) {
    return TestHelpers::build_horizons_for_basic_control_systems<
        element_component>(time, runner, position_function, coord_map);
  };

  // Run the actual control system test.
  system_helper.run_control_system_test(runner, final_time, make_not_null(&gen),
                                        horizon_function);

  // Grab results
  std::array<double, 3> grid_position_of_a;
  std::array<double, 3> grid_position_of_b;
  std::tie(grid_position_of_a, grid_position_of_b) =
      TestHelpers::grid_frame_horizon_centers_for_basic_control_systems<
          element_component>(final_time, runner, position_function, coord_map);

  // Our expected positions are just the initial positions
  const std::array<double, 3> expected_grid_position_of_a{
      {0.5 * initial_separation, 0.0, 0.0}};
  const std::array<double, 3> expected_grid_position_of_b{
      {-0.5 * initial_separation, 0.0, 0.0}};

  const auto& rotation_f_of_t = dynamic_cast<
      domain::FunctionsOfTime::QuaternionFunctionOfTime<RotationDerivOrder>&>(
      *functions_of_time.at(rotation_name));
  const auto& translation_f_of_t = dynamic_cast<
      domain::FunctionsOfTime::PiecewisePolynomial<TranslationDerivOrder>&>(
      *functions_of_time.at(translation_name));
  const auto& expansion_f_of_t = dynamic_cast<
      domain::FunctionsOfTime::PiecewisePolynomial<ExpansionDerivOrder>&>(
      *functions_of_time.at(expansion_name));

  const auto omega = rotation_f_of_t.angle_func_and_deriv(final_time)[1];
  const auto trans_and_2_derivs =
      translation_f_of_t.func_and_2_derivs(final_time);
  const auto expansion_factor = expansion_f_of_t.func(final_time)[0][0];

  // The control system gets more accurate the longer you run for. This is
  // the accuracy we can achieve in this amount of time.
  Approx custom_approx1 = Approx::custom().epsilon(5.0e-5).scale(1.0);
  Approx custom_approx2 = Approx::custom().epsilon(rotation_eps).scale(1.0);

  const DataVector expected_omega{
      {0.0, 0.0, binary_trajectories.angular_velocity(final_time)}};
  const DataVector expected_translation_2nd_deriv{3, 0.0};
  CHECK_ITERABLE_CUSTOM_APPROX(expected_omega, omega, custom_approx1);
  CHECK_ITERABLE_CUSTOM_APPROX(trans_and_2_derivs[2],
                               expected_translation_2nd_deriv, custom_approx1);
  CHECK_ITERABLE_CUSTOM_APPROX(trans_and_2_derivs[1],
                               array_to_datavector(initial_velocity),
                               custom_approx1);
  CHECK_ITERABLE_CUSTOM_APPROX(
      trans_and_2_derivs[0], array_to_datavector(initial_velocity * final_time),
      custom_approx1);
  CHECK_ITERABLE_CUSTOM_APPROX(
      binary_trajectories.separation(final_time) / initial_separation,
      expansion_factor, custom_approx1);

  CHECK_ITERABLE_CUSTOM_APPROX(expected_grid_position_of_a, grid_position_of_a,
                               custom_approx2);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_grid_position_of_b, grid_position_of_b,
                               custom_approx2);
}
}  // namespace

// Currently the test takes a long time because the logic for the control system
// isn't the most optimal. This should be fixed in the near future.
// [[TimeOut, 25]]
SPECTRE_TEST_CASE("Unit.ControlSystem.Systems.RotScaleTrans",
                  "[ControlSystem][Unit]") {
  // For rotation deriv order 2, we need a different epsilon because controlling
  // the first derivative of omega (second derivative of the angle) doesn't give
  // as accurate results for the final positions of the objects in the grid
  // frame. This is also done is Systems/Test_Rotation.
  test_rotscaletrans_control_system<2, 2, 2>(5.0e-3);
  test_rotscaletrans_control_system<3, 2, 2>(5.0e-3);
  test_rotscaletrans_control_system<2, 3, 2>();
  test_rotscaletrans_control_system<2, 2, 3>(5.0e-3);
  test_rotscaletrans_control_system<3, 3, 2>();
  test_rotscaletrans_control_system<2, 3, 3>();
  test_rotscaletrans_control_system<3, 3, 3>();
}
}  // namespace control_system
