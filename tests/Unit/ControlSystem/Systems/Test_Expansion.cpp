// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <tuple>

#include "ControlSystem/Systems/Expansion.hpp"
#include "ControlSystem/Tags/MeasurementTimescales.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/TimeDependent/CubicScale.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/ControlSystem/SystemHelpers.hpp"
#include "Helpers/PointwiseFunctions/PostNewtonian/BinaryTrajectories.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Frame {
struct Distorted;
struct Inertial;
}  // namespace Frame

namespace control_system {
namespace {
using ExpansionMap = domain::CoordinateMaps::TimeDependent::CubicScale<3>;

using CoordMap =
    domain::CoordinateMap<Frame::Distorted, Frame::Inertial, ExpansionMap>;

template <size_t DerivOrder>
void test_expansion_control_system() {
  using metavars = TestHelpers::MockMetavars<0, 0, DerivOrder, 0>;
  using expansion_component = typename metavars::expansion_component;
  using element_component = typename metavars::element_component;
  using observer_component = typename metavars::observer_component;
  using expansion_system = typename metavars::expansion_system;
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
      "DomainCreator:\n"
      "  FakeCreator:\n"
      "    NumberOfExcisions: 2\n"
      "    NumberOfComponents:\n"
      "      Expansion: 1\n"
      "ControlSystems:\n"
      "  WriteDataToDisk: false\n"
      "  MeasurementsPerUpdate: 4\n"
      "  Expansion:\n"
      "    IsActive: true\n"
      "    Averager:\n"
      "      AverageTimescaleFraction: 0.25\n"
      "      Average0thDeriv: true\n"
      "    Controller:\n"
      "      UpdateFraction: 0.3\n"
      "    TimescaleTuner:\n"
      "      InitialTimescales: [0.5]\n"
      "      MinTimescale: 0.1\n"
      "      MaxTimescale: 10.\n"
      "      DecreaseThreshold: 2.0\n"
      "      IncreaseThreshold: 0.1\n"
      "      IncreaseFactor: 1.01\n"
      "      DecreaseFactor: 0.99\n"
      "    ControlError:\n";

  // Initialize everything within the system helper
  system_helper.setup_control_system_test(
      initial_time, initial_separation, input_options,
      TestHelpers::initialize_expansion_functions_of_time<expansion_system>);

  // Get references to everything that was set up inside the system helper. The
  // domain and two functions of time are not const references because they need
  // to be moved into the runner
  auto& domain = system_helper.domain();
  auto& initial_functions_of_time = system_helper.initial_functions_of_time();
  auto& initial_measurement_timescales =
      system_helper.initial_measurement_timescales();
  const auto& init_exp_tuple =
      system_helper.template init_tuple<expansion_system>();

  auto grid_center_A = domain.excision_spheres().at("ExcisionSphereA").center();
  auto grid_center_B = domain.excision_spheres().at("ExcisionSphereB").center();

  // Setup runner and all components
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  MockRuntimeSystem runner{
      {"DummyFileName", std::move(domain), 4, false, ::Verbosity::Silent,
       std::move(grid_center_A), std::move(grid_center_B)},
      {std::move(initial_functions_of_time),
       std::move(initial_measurement_timescales)}};
  ActionTesting::emplace_singleton_component_and_initialize<
      expansion_component>(make_not_null(&runner), ActionTesting::NodeId{0},
                           ActionTesting::LocalCoreId{0}, init_exp_tuple);
  ActionTesting::emplace_array_component<element_component>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0}, 0);
  ActionTesting::emplace_nodegroup_component<observer_component>(
      make_not_null(&runner));

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  const BinaryTrajectories binary_trajectories{initial_separation};

  const std::string expansion_name =
      system_helper.template name<expansion_system>();

  // Create coordinate maps for mapping the PN expansion to the "grid" frame
  // where the control system does its calculations
  // The outer boundary is at 1000.0 so that we don't have to worry about it.
  ExpansionMap expansion_map{1000.0, expansion_name,
                             expansion_name + "OuterBoundary"s};

  CoordMap coord_map{expansion_map};

  // Get the functions of time from the cache to use in the maps
  const auto& cache = ActionTesting::cache<element_component>(runner, 0);
  const auto& functions_of_time =
      Parallel::get<domain::Tags::FunctionsOfTime>(cache);

  const auto position_function = [&binary_trajectories](const double time) {
    const double separation = binary_trajectories.separation(time);
    return std::pair<std::array<double, 3>, std::array<double, 3>>{
        {0.5 * separation, 0.0, 0.0}, {-0.5 * separation, 0.0, 0.0}};
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

  const auto& expansion_f_of_t =
      dynamic_cast<domain::FunctionsOfTime::PiecewisePolynomial<DerivOrder>&>(
          *functions_of_time.at(expansion_name));

  const double exp_factor = expansion_f_of_t.func(final_time)[0][0];

  // The control system gets more accurate the longer you run for. However, this
  // is the floor of our accuracy for a changing function of time.
  Approx custom_approx = Approx::custom().epsilon(5.0e-6).scale(1.0);
  const double expected_exp_factor =
      binary_trajectories.separation(final_time) / initial_separation;
  CHECK(custom_approx(expected_exp_factor) == exp_factor);

  CHECK_ITERABLE_CUSTOM_APPROX(expected_grid_position_of_a, grid_position_of_a,
                               custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_grid_position_of_b, grid_position_of_b,
                               custom_approx);
}

void test_names() {
  using expansion = control_system::Systems::Expansion<
      2, control_system::measurements::BothHorizons>;

  CHECK(pretty_type::name<expansion>() == "Expansion");
  CHECK(*expansion::component_name(0, 1) == "Expansion");
  CHECK(*expansion::component_name(1, 1) == "Expansion");

#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      ([]() {
        const std::string component_name = *expansion::component_name(1, 2);
        (void)component_name;
      })(),
      Catch::Contains(
          "Expansion control expects 1 component but there are 2 instead."));
#endif  // SPECTRE_DEBUG
}

SPECTRE_TEST_CASE("Unit.ControlSystem.Systems.Expansion",
                  "[ControlSystem][Unit]") {
  test_names();
  test_expansion_control_system<2>();
  test_expansion_control_system<3>();
}
}  // namespace
}  // namespace control_system
