// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "ControlSystem/ControlErrors/Expansion.hpp"
#include "ControlSystem/ControlErrors/Rotation.hpp"
#include "ControlSystem/ControlErrors/Translation.hpp"
#include "ControlSystem/Tags/MeasurementTimescales.hpp"
#include "ControlSystem/Tags/QueueTags.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "ControlSystem/UpdateFunctionOfTime.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/ControlSystem/SystemHelpers.hpp"
#include "Parallel/GlobalCache.hpp"
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
  auto& is_active_map = system_helper.is_active_map();
  auto& initial_functions_of_time = system_helper.initial_functions_of_time();
  auto& initial_measurement_timescales =
      system_helper.initial_measurement_timescales();
  auto system_to_combined_names = system_helper.system_to_combined_names();
  const std::string translation_name =
      system_helper.template name<translation_system>();

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
  const DataVector grid_A{{initial_separation / 2.0, 0.0, 0.0}};
  const DataVector grid_B{{-initial_separation / 2.0, 0.0, 0.0}};
  QueueTuple fake_measurement_tuple{pos_A, pos_B};

  const DataVector grid_position_average = 0.5 * (grid_A + grid_B);
  const DataVector current_position_average = 0.5 * (pos_A + pos_B);

  const DataVector grid_separation = grid_A - grid_B;
  const DataVector current_separation = pos_A - pos_B;

  double current_separation_dot_grid_separation = 0.0;
  double current_separation_dot_grid_average = 0.0;
  double grid_separation_dot_grid_average = 0.0;
  double grid_separation_dot_grid_separation = 0.0;
  for (size_t i = 0; i < 3; i++) {
    current_separation_dot_grid_separation +=
        current_separation[i] * grid_separation[i];
    current_separation_dot_grid_average +=
        current_separation[i] * grid_position_average[i];
    grid_separation_dot_grid_average +=
        grid_separation[i] * grid_position_average[i];
    grid_separation_dot_grid_separation +=
        grid_separation[i] * grid_separation[i];
  }

  using ControlError = translation_system::control_error;

  // This is before the first expiration time
  const double check_time = 0.1;
  const DataVector control_error =
      ControlError{}(::TimescaleTuner<true>{}, cache, check_time,
                     translation_name, fake_measurement_tuple);

  // The quaternion should be the unit quaternion (1,0,0,0) which means the
  // quaternion multiplication in the translation control error is the identity
  // so we avoid actually doing quaternion multiplication. Also the expansion
  // factor should be 1.0 so we don't have to multiply/divide by that where we
  // normally would have
  const DataVector expected_control_error =
      (grid_separation_dot_grid_separation * current_position_average -
       current_separation_dot_grid_separation * grid_position_average -
       grid_separation_dot_grid_average * current_separation +
       current_separation_dot_grid_average * grid_separation) /
      grid_separation_dot_grid_separation;
  ;

  Approx custom_approx = Approx::custom().epsilon(1.0e-14).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(control_error, expected_control_error,
                               custom_approx);
}

struct SingleMetavars {
  using const_global_cache_tags =
      tmpl::list<domain::Tags::FunctionsOfTimeInitialize>;
  using component_list = tmpl::list<>;
};

void test_single_object_translation_control_error() {
  using ControlError = ControlErrors::Translation<1>;
  using QueueTag =
      control_system::QueueTags::Center<::domain::ObjectLabel::None>;
  using QueueTuple = tuples::TaggedTuple<QueueTag>;
  using CacheType = Parallel::GlobalCache<SingleMetavars>;

  const TimescaleTuner<true> unused_tuner{};
  // Along x-axis makes things easier to test
  const QueueTuple fake_measurement_tuple{DataVector{-0.2, 0.0, 0.0}};

  const double check_time = 0.1;
  const std::string function_of_time_name{"Translation"};

  {
    const CacheType cache{{domain::FunctionsOfTimeMap{}}};

    const DataVector control_error =
        ControlError{}(unused_tuner, cache, check_time, function_of_time_name,
                       fake_measurement_tuple);
    const DataVector& expected_control_error =
        tuples::get<QueueTag>(fake_measurement_tuple);
    CHECK(control_error == expected_control_error);
  }

  {
    domain::FunctionsOfTimeMap functions_of_time{};
    functions_of_time["Expansion"] =
        std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<0>>(
            0.0, std::array{DataVector{0.5}}, check_time * 10.0);
    const CacheType cache{{std::move(functions_of_time)}};

    const DataVector control_error =
        ControlError{}(unused_tuner, cache, check_time, function_of_time_name,
                       fake_measurement_tuple);
    const DataVector expected_control_error =
        0.5 * tuples::get<QueueTag>(fake_measurement_tuple);
    CHECK(control_error == expected_control_error);
  }

  {
    domain::FunctionsOfTimeMap functions_of_time{};
    functions_of_time["Expansion"] =
        std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<0>>(
            0.0, std::array{DataVector{0.5}}, check_time * 10.0);
    functions_of_time["Rotation"] =
        std::make_unique<domain::FunctionsOfTime::QuaternionFunctionOfTime<2>>(
            0.0, std::array{DataVector{1.0, 0.0, 0.0, 0.0}},
            std::array{DataVector{3, 0.0}, DataVector{0.0, 0.0, 1.5},
                       DataVector{3, 0.0}},
            check_time * 10.0);
    const CacheType cache{{std::move(functions_of_time)}};

    const DataVector control_error =
        ControlError{}(unused_tuner, cache, check_time, function_of_time_name,
                       fake_measurement_tuple);
    // Rotation and expand
    const DataVector expected_control_error =
        0.5 * -0.2 * DataVector{cos(0.1 * 1.5), sin(0.1 * 1.5), 0.0};
    CHECK_ITERABLE_APPROX(control_error, expected_control_error);
  }
}

SPECTRE_TEST_CASE("Unit.ControlSystem.ControlErrors.Translation",
                  "[ControlSystem][Unit]") {
  test_translation_control_error();
  test_single_object_translation_control_error();
}
}  // namespace
}  // namespace control_system
