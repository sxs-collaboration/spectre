// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <utility>

#include "ControlSystem/ControlErrors/Expansion.hpp"
#include "ControlSystem/ControlErrors/Rotation.hpp"
#include "ControlSystem/DataVectorHelpers.hpp"
#include "ControlSystem/Tags/MeasurementTimescales.hpp"
#include "ControlSystem/Tags/QueueTags.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/ControlSystem/SystemHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace control_system {
namespace {
void test_translation_control_error() {
  // Since we are only doing translation, turn off the
  // other control systems by passing 0 for their deriv orders
  constexpr size_t deriv_order = 2;
  using metavars = TestHelpers::MockMetavars<deriv_order, 0, 0, 0>;
  using element_component = typename metavars::element_component;
  using translation_system = typename metavars::translation_system;

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
      "    NumberOfExcisions: 2\n"
      "    NumberOfComponents:\n"
      "      Translation: 3\n"
      "ControlSystems:\n"
      "  WriteDataToDisk: false\n"
      "  MeasurementsPerUpdate: 4\n"
      "  Translation:\n"
      "    IsActive: true\n"
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
  system_helper.setup_control_system_test(
      initial_time, initial_separation, input_options,
      TestHelpers::initialize_translation_functions_of_time<
          translation_system>);

  // Get references to everything that was set up inside the system helper. The
  // domain and two functions of time are not const references because they need
  // to be moved into the runner
  auto& domain = system_helper.domain();
  auto& initial_functions_of_time = system_helper.initial_functions_of_time();
  auto& initial_measurement_timescales =
      system_helper.initial_measurement_timescales();
  const std::string translation_name =
      system_helper.template name<translation_system>();

  auto grid_center_A = domain.excision_spheres().at("ExcisionSphereA").center();
  auto grid_center_B = domain.excision_spheres().at("ExcisionSphereB").center();

  // Setup runner and element component because it's the easiest way to get the
  // global cache
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  MockRuntimeSystem runner{
      {"DummyFileName", std::move(domain), 4, false, ::Verbosity::Silent,
       std::move(grid_center_A), std::move(grid_center_B)},
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
  const DataVector grid_A{{initial_separation / 2.0, 0.0, 0.0}};
  QueueTuple fake_measurement_tuple{pos_A, pos_B};

  using ControlError = translation_system::control_error;

  // This is before the first expiration time
  const double check_time = 0.1;
  const DataVector control_error = ControlError{}(
      cache, check_time, translation_name, fake_measurement_tuple);

  // Calculated errors from other basic control systems
  const DataVector rotation_control_error =
      DataVector{{0.0, -15.0, 105.0}} / 75.0;
  const double expansion_control_error =
      (pos_A[0] - pos_B[0]) / initial_separation - 1.0;

  const DataVector rot_control_err_cross_grid =
      cross(rotation_control_error, grid_A);

  // The quaternion should be the unit quaternion (1,0,0,0) which means the
  // quaternion multiplication in the translation control error is the identity
  // so we avoid actually doing quaternion multiplication. Also the expansion
  // factor should be 1.0 so we don't have to multiply/divide by that where we
  // normally would have
  const DataVector expected_control_error = pos_A - grid_A -
                                            rot_control_err_cross_grid -
                                            expansion_control_error * grid_A;

  Approx custom_approx = Approx::custom().epsilon(1.0e-14).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(control_error, expected_control_error,
                               custom_approx);
}

SPECTRE_TEST_CASE("Unit.ControlSystem.ControlErrors.Translation",
                  "[ControlSystem][Unit]") {
  test_translation_control_error();
}
}  // namespace
}  // namespace control_system
