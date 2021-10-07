// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "ControlSystem/Systems/Rotation.hpp"
#include "ControlSystem/Tags.hpp"
#include "ControlSystem/Tags/MeasurementTimescales.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/TimeDependent/Rotation.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/ControlSystem/SystemHelpers.hpp"
#include "Helpers/PointwiseFunctions/PostNewtonian/BinaryTrajectories.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Frame {
struct Grid;
struct Inertial;
}  // namespace Frame

namespace control_system {
namespace {
using RotationMap = domain::CoordinateMaps::TimeDependent::Rotation<3>;

using CoordMap =
    domain::CoordinateMap<Frame::Grid, Frame::Inertial, RotationMap>;

template <size_t DerivOrder>
void test_rotation_control_system(const bool newtonian) {
  // Since we are only doing rotation, turn off the
  // other control systems by passing 0 for their deriv orders
  using metavars = TestHelpers::MockMetavars<DerivOrder, 0>;
  using rotation_component = typename metavars::rotation_component;
  using element_component = typename metavars::element_component;
  MAKE_GENERATOR(gen);

  // Global things
  domain::FunctionsOfTime::register_derived_with_charm();
  const double initial_time = 0.0;
  const double initial_separation = 15.0;
  // This final time is chosen so that the damping timescales have adequate time
  // to reach the maximum damping timescale
  const double final_time = 500.0;

  // Set up the system helper.
  control_system::TestHelpers::SystemHelper<metavars> system_helper{};

  const std::string input_options =
      "Evolution:\n"
      "  InitialTime: 0.0\n"
      "ControlSystems:\n"
      "  WriteDataToDisk: false\n"
      "  Rotation:\n"
      "    Averager:\n"
      "      AverageTimescaleFraction: 0.25\n"
      "      Average0thDeriv: true\n"
      "    Controller:\n"
      "      UpdateFraction: 0.3\n"
      "    TimescaleTuner:\n"
      "      InitialTimescales: [0.5, 0.5, 0.5]\n"
      "      MinTimescale: 0.1\n"
      "      MaxTimescale: 10.\n"
      "      DecreaseThreshold: 2.0\n"
      "      IncreaseThreshold: 0.1\n"
      "      IncreaseFactor: 1.01\n"
      "      DecreaseFactor: 0.99\n"
      "    ControlError:\n";

  // Initialize everything within the system helper
  system_helper.setup_control_system_test(initial_time, initial_separation,
                                          input_options);

  // Get references to everything that was set up inside the system helper. The
  // domain and two functions of time are not const references because they need
  // to be moved into the runner
  auto& domain = system_helper.domain();
  auto& initial_functions_of_time = system_helper.initial_functions_of_time();
  auto& initial_measurement_timescales =
      system_helper.initial_measurement_timescales();
  const auto& init_rot_tuple = system_helper.init_rot_tuple();

  // Setup runner and all components
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  MockRuntimeSystem runner{{std::move(domain)},
                           {std::move(initial_functions_of_time),
                            std::move(initial_measurement_timescales)}};
  ActionTesting::emplace_singleton_component_and_initialize<rotation_component>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0}, init_rot_tuple);
  ActionTesting::emplace_array_component<element_component>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0}, 0);

  ActionTesting::set_phase(make_not_null(&runner), metavars::Phase::Testing);

  const BinaryTrajectories binary_trajectories{
      initial_separation, {0.0, 0.0, 0.0}, newtonian};

  const std::string& rotation_name = system_helper.rotation_name();

  // Create coordinate map for mapping the PN rotation to the "grid" frame
  // where the control system does its calculations
  RotationMap rotation_map{rotation_name};

  CoordMap coord_map{rotation_map};

  // Get the functions of time from the cache to use in the maps
  const auto& cache = ActionTesting::cache<element_component>(runner, 0);
  const auto& functions_of_time =
      Parallel::get<domain::Tags::FunctionsOfTime>(cache);

  const auto position_function = [&binary_trajectories](const double time) {
    return binary_trajectories.positions_no_expansion(time);
  };

  // Run the actual control system test.
  system_helper.run_control_system_test(runner, final_time, make_not_null(&gen),
                                        position_function, coord_map);

  // Grab results
  const std::array<double, 3> grid_position_of_a =
      system_helper.grid_position_of_a();
  const std::array<double, 3> grid_position_of_b =
      system_helper.grid_position_of_b();

  // Our expected positions are just the initial positions
  const std::array<double, 3> expected_grid_position_of_a{
      {-0.5 * initial_separation, 0.0, 0.0}};
  const std::array<double, 3> expected_grid_position_of_b{
      {0.5 * initial_separation, 0.0, 0.0}};

  const auto& rotation_f_of_t = dynamic_cast<
      domain::FunctionsOfTime::QuaternionFunctionOfTime<DerivOrder>&>(
      *functions_of_time.at(rotation_name));

  const auto omega = rotation_f_of_t.angle_func_and_deriv(final_time)[1];

  // The control system gets more accurate the longer you run for. This is
  // the accuracy we can achieve in this amount of time. For PN deriv order 2,
  // we need a different epsilon because controlling the first derivative of
  // omega (second derivative of the angle) doesn't give as accurate results for
  // the final positions of the objects in the grid frame.
  double eps = 5.0e-5;
  if (DerivOrder == 2 and not newtonian) {
    eps = 5.0e-3;
  }
  Approx custom_approx1 = Approx::custom().epsilon(5.0e-5).scale(1.0);
  Approx custom_approx2 = Approx::custom().epsilon(eps).scale(1.0);
  const DataVector expected_omega{
      {0.0, 0.0, binary_trajectories.angular_velocity(final_time)}};
  CHECK_ITERABLE_CUSTOM_APPROX(expected_omega, omega, custom_approx1);

  CHECK_ITERABLE_CUSTOM_APPROX(expected_grid_position_of_a, grid_position_of_a,
                               custom_approx2);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_grid_position_of_b, grid_position_of_b,
                               custom_approx2);
}

void test_names() {
  using rotation = control_system::Systems::Rotation<2>;

  CHECK(pretty_type::name<rotation>() == "Rotation");
  CHECK(rotation::component_name(0) == "x");
  CHECK(rotation::component_name(1) == "y");
  CHECK(rotation::component_name(2) == "z");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ControlSystem.Systems.Rotation",
                  "[ControlSystem][Unit]") {
  test_names();
  test_rotation_control_system<2>(true);
  test_rotation_control_system<2>(false);
  test_rotation_control_system<3>(true);
  test_rotation_control_system<3>(false);
}
}  // namespace control_system
