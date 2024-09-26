// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "ControlSystem/Tags/MeasurementTimescales.hpp"
#include "ControlSystem/Tags/QueueTags.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/ControlSystem/SystemHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace control_system {
namespace {
void test_rotation_control_error() {
  // Since we are only doing rotation, turn off the
  // other control systems by passing 0 for their deriv orders
  constexpr size_t deriv_order = 2;
  using metavars = TestHelpers::MockMetavars<0, deriv_order, 0, 0>;
  using element_component = typename metavars::element_component;
  using rotation_system = typename metavars::rotation_system;
  MAKE_GENERATOR(gen);

  // Global things
  domain::FunctionsOfTime::register_derived_with_charm();
  const double initial_time = 0.0;
  const double initial_separation = 15.0;

  // Set up the system helper.
  control_system::TestHelpers::SystemHelper<metavars> system_helper{};

  const std::string input_options =
      "Evolution:\n"
      "  InitialTime: 0.0\n"
      "DomainCreator:\n"
      "  FakeCreator:\n"
      "    NumberOfComponents:\n"
      "      Rotation: 3\n"
      "ControlSystems:\n"
      "  WriteDataToDisk: false\n"
      "  MeasurementsPerUpdate: 4\n"
      "  Rotation:\n"
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
      "    ControlError:\n";

  // Initialize everything within the system helper
  system_helper.setup_control_system_test(
      initial_time, initial_separation, input_options,
      TestHelpers::initialize_rotation_functions_of_time<rotation_system>);

  // Get references to everything that was set up inside the system helper. The
  // domain and two functions of time are not const references because they need
  // to be moved into the runner
  auto& domain = system_helper.domain();
  auto& is_active_map = system_helper.is_active_map();
  auto& initial_functions_of_time = system_helper.initial_functions_of_time();
  auto& initial_measurement_timescales =
      system_helper.initial_measurement_timescales();
  auto system_to_combined_names = system_helper.system_to_combined_names();
  const std::string rotation_name =
      system_helper.template name<rotation_system>();

  auto grid_center_A = domain.excision_spheres().at("ExcisionSphereA").center();
  auto grid_center_B = domain.excision_spheres().at("ExcisionSphereB").center();

  // Setup runner and element component because it's the easiest way to get the
  // global cache
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  MockRuntimeSystem runner{
      {"DummyFileName", std::move(domain), 4, false, ::Verbosity::Silent,
       std::move(is_active_map), std::move(grid_center_A),
       std::move(grid_center_B), std::move(system_to_combined_names)},
      {std::move(initial_functions_of_time),
       std::move(initial_measurement_timescales)}};
  ActionTesting::emplace_array_component<element_component>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0}, 0);
  const auto& cache = ActionTesting::cache<element_component>(runner, 0);

  using QueueTuple = tuples::TaggedTuple<
      control_system::QueueTags::Center<::domain::ObjectLabel::A>,
      control_system::QueueTags::Center<::domain::ObjectLabel::B>>;

  // Create fake measurements.
  const DataVector pos_A{{2.0, 3.0, 6.0}};
  const DataVector pos_B{{-3.0, -4.0, 5.0}};
  QueueTuple fake_measurement_tuple{pos_A, pos_B};

  using ControlError = rotation_system::control_error;

  // This is before the first expiration time
  const double check_time = 0.1;
  const DataVector control_error =
      ControlError{}(::TimescaleTuner<true>{}, cache, check_time, rotation_name,
                     fake_measurement_tuple);

  // Calculated error = (grid_diff cross pos_diff) / (grid_diff dot pos_diff) by
  // hand
  const DataVector expected_control_error =
      DataVector{{0.0, -15.0, 105.0}} / 75.0;

  CHECK_ITERABLE_APPROX(control_error, expected_control_error);
}

SPECTRE_TEST_CASE("Unit.ControlSystem.ControlErrors.Rotation",
                  "[ControlSystem][Unit]") {
  test_rotation_control_error();
}
}  // namespace
}  // namespace control_system
