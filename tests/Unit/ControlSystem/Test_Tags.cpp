// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "ControlSystem/Averager.hpp"
#include "ControlSystem/Component.hpp"
#include "ControlSystem/Controller.hpp"
#include "ControlSystem/Tags/FunctionsOfTimeInitialize.hpp"
#include "ControlSystem/Tags/IsActiveMap.hpp"
#include "ControlSystem/Tags/MeasurementTimescales.hpp"
#include "ControlSystem/Tags/OptionTags.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/ControlSystem/SystemHelpers.hpp"
#include "Helpers/ControlSystem/TestStructs.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

namespace {
struct LabelA {};
struct LabelB {};
struct Rotation {};
using system = control_system::TestHelpers::System<
    2, LabelA, control_system::TestHelpers::Measurement<LabelA>>;
using system2 = control_system::TestHelpers::System<
    2, LabelB, control_system::TestHelpers::Measurement<LabelA>>;
using quat_system = control_system::TestHelpers::System<
    2, Rotation, control_system::TestHelpers::Measurement<Rotation>>;

struct MetavarsEmpty {
  static constexpr size_t volume_dim = 3;
};

using FakeCreator = control_system::TestHelpers::FakeCreator;

void test_all_tags() {
  INFO("Test all tags");
  using write_tag = control_system::Tags::WriteDataToDisk;
  TestHelpers::db::test_simple_tag<write_tag>("WriteDataToDisk");
  using observe_tag = control_system::Tags::ObserveCenters;
  TestHelpers::db::test_simple_tag<observe_tag>("ObserveCenters");
  using averager_tag = control_system::Tags::Averager<system>;
  TestHelpers::db::test_simple_tag<averager_tag>("Averager");
  using timescaletuner_tag = control_system::Tags::TimescaleTuner<system>;
  TestHelpers::db::test_simple_tag<timescaletuner_tag>("TimescaleTuner");
  using controller_tag = control_system::Tags::Controller<system>;
  TestHelpers::db::test_simple_tag<controller_tag>("Controller");
  using fot_tag = control_system::Tags::FunctionsOfTimeInitialize;
  TestHelpers::db::test_simple_tag<fot_tag>("FunctionsOfTime");
  using verbosity_tag = control_system::Tags::Verbosity;
  TestHelpers::db::test_simple_tag<verbosity_tag>("Verbosity");

  using control_error_tag = control_system::Tags::ControlError<system>;
  TestHelpers::db::test_simple_tag<control_error_tag>("ControlError");

  using active_tag = control_system::Tags::IsActiveMap;
  TestHelpers::db::test_simple_tag<active_tag>("IsActiveMap");

  using measurement_tag = control_system::Tags::MeasurementTimescales;
  TestHelpers::db::test_simple_tag<measurement_tag>("MeasurementTimescales");

  using measurements_per_update_tag =
      control_system::Tags::MeasurementsPerUpdate;
  TestHelpers::db::test_simple_tag<measurements_per_update_tag>(
      "MeasurementsPerUpdate");
  using current_measurement_tag =
      control_system::Tags::CurrentNumberOfMeasurements;
  TestHelpers::db::test_simple_tag<current_measurement_tag>(
      "CurrentNumberOfMeasurements");
  using system_to_combined_tag = control_system::Tags::SystemToCombinedNames;
  TestHelpers::db::test_simple_tag<system_to_combined_tag>(
      "SystemToCombinedNames");
  using aggregators_tag = control_system::Tags::UpdateAggregators;
  TestHelpers::db::test_simple_tag<aggregators_tag>("UpdateAggregators");
}

void test_control_sys_inputs() {
  INFO("Test control system inputs");
  const double decrease_timescale_threshold = 1.0e-2;
  const double increase_timescale_threshold = 1.0e-4;
  const double increase_factor = 1.01;
  const double decrease_factor = 0.99;
  const double max_timescale = 10.0;
  const double min_timescale = 1.0e-3;
  const TimescaleTuner<true> expected_tuner(
      std::vector<double>{1.}, max_timescale, min_timescale,
      increase_timescale_threshold, increase_factor,
      decrease_timescale_threshold, decrease_factor);
  const Averager<1> expected_averager(0.25, true);
  const Controller<2> expected_controller(0.3);
  const std::string expected_name{"LabelA"};

  const auto input_holder = TestHelpers::test_option_tag<
      control_system::OptionTags::ControlSystemInputs<system>>(
      "IsActive: false\n"
      "Averager:\n"
      "  AverageTimescaleFraction: 0.25\n"
      "  Average0thDeriv: true\n"
      "Controller:\n"
      "  UpdateFraction: 0.3\n"
      "TimescaleTuner:\n"
      "  InitialTimescales: [1.]\n"
      "  MinTimescale: 1e-3\n"
      "  MaxTimescale: 10.\n"
      "  DecreaseThreshold: 1e-2\n"
      "  IncreaseThreshold: 1e-4\n"
      "  IncreaseFactor: 1.01\n"
      "  DecreaseFactor: 0.99\n"
      "ControlError:\n");
  CHECK_FALSE(input_holder.is_active);
  CHECK(expected_averager == input_holder.averager);
  CHECK(expected_controller == input_holder.controller);
  CHECK(expected_tuner == input_holder.tuner);
  CHECK(expected_name ==
        std::decay_t<decltype(input_holder)>::control_system::name());

  const auto write_data =
      TestHelpers::test_option_tag<control_system::OptionTags::WriteDataToDisk>(
          "true");
  CHECK(write_data);
  // We don't check the control error because the example one is empty and
  // doesn't have a comparison operator. Once a control error is added that
  // contains member data (and thus, options), then it can be tested
}

void test_individual_tags() {
  const auto create_expected_tuner = [](const size_t num_components) {
    const double decrease_timescale_threshold = 1.0e-2;
    const double increase_timescale_threshold = 1.0e-4;
    const double increase_factor = 1.01;
    const double decrease_factor = 0.99;
    const double max_timescale = 10.0;
    const double min_timescale = 1.0e-3;
    return TimescaleTuner<true>{std::vector<double>(num_components, 1.0),
                                max_timescale,
                                min_timescale,
                                increase_timescale_threshold,
                                increase_factor,
                                decrease_timescale_threshold,
                                decrease_factor};
  };

  const auto tuner_str = [](const bool is_active) -> std::string {
    return "IsActive: " + (is_active ? "true"s : "false"s) +
           "\n"
           "Averager:\n"
           "  AverageTimescaleFraction: 0.25\n"
           "  Average0thDeriv: true\n"
           "Controller:\n"
           "  UpdateFraction: 0.3\n"
           "TimescaleTuner:\n"
           "  InitialTimescales: 1.\n"
           "  MinTimescale: 1e-3\n"
           "  MaxTimescale: 10.\n"
           "  DecreaseThreshold: 1e-2\n"
           "  IncreaseThreshold: 1e-4\n"
           "  IncreaseFactor: 1.01\n"
           "  DecreaseFactor: 0.99\n"
           "ControlError:\n";
  };

  using tuner_tag = control_system::Tags::TimescaleTuner<system>;
  using quat_tuner_tag = control_system::Tags::TimescaleTuner<quat_system>;

  const auto holder = TestHelpers::test_option_tag<
      control_system::OptionTags::ControlSystemInputs<system>>(tuner_str(true));
  const auto holder2 = TestHelpers::test_option_tag<
      control_system::OptionTags::ControlSystemInputs<system2>>(
      tuner_str(true));
  const auto quat_holder = TestHelpers::test_option_tag<
      control_system::OptionTags::ControlSystemInputs<quat_system>>(
      tuner_str(true));
  const auto inactive_holder = TestHelpers::test_option_tag<
      control_system::OptionTags::ControlSystemInputs<system>>(
      tuner_str(false));

  const std::unique_ptr<DomainCreator<3>> creator =
      std::make_unique<FakeCreator>(std::unordered_map<std::string, size_t>{
          {system::name(), 2}, {quat_system::name(), 3}});
  const std::unique_ptr<DomainCreator<3>> creator_empty =
      std::make_unique<FakeCreator>(std::unordered_map<std::string, size_t>{});

  const TimescaleTuner<true> created_tuner =
      tuner_tag::create_from_options<MetavarsEmpty>(holder, creator, 0.0);
  const TimescaleTuner<true> quat_created_tuner =
      quat_tuner_tag::create_from_options<MetavarsEmpty>(quat_holder, creator,
                                                         0.0);
  const TimescaleTuner<true> inactive_created_tuner =
      tuner_tag::create_from_options<MetavarsEmpty>(inactive_holder,
                                                    creator_empty, 0.0);

  CHECK(created_tuner == create_expected_tuner(2));
  CHECK(quat_created_tuner == create_expected_tuner(3));
  CHECK(inactive_created_tuner == create_expected_tuner(1));

  using control_error_tag = control_system::Tags::ControlError<system>;
  using control_error_tag2 = control_system::Tags::ControlError<system2>;

  // We're only checking for errors here, so the fact that it doesn't error is
  // enough of a check. We test errors below.
  [[maybe_unused]] const control_error_tag::type created_control_error =
      control_error_tag::create_from_options(holder);
  [[maybe_unused]] const control_error_tag2::type created_control_error2 =
      control_error_tag2::create_from_options(holder2);

  using controller_tag = control_system::Tags::Controller<system>;

  Controller<2> expected_controller(0.3);
  expected_controller.set_initial_update_time(0.0);
  expected_controller.assign_time_between_updates(
      min(created_tuner.current_timescale()));

  const Controller<2> created_controller =
      controller_tag::create_from_options<MetavarsEmpty>(holder, creator, 0.0);

  CHECK(created_controller == expected_controller);

  using active_map_tag = control_system::Tags::IsActiveMap;

  const auto active_map = active_map_tag::create_from_options<MetavarsEmpty>(
      quat_holder, holder2, inactive_holder);
  CHECK(active_map == std::unordered_map<std::string, bool>{{"Rotation", true},
                                                            {"LabelB", true},
                                                            {"LabelA", false}});
}

struct NamesMetavars {
  using component_list =
      tmpl::list<ControlComponent<NamesMetavars, system>,
                 ControlComponent<NamesMetavars, system2>,
                 ControlComponent<NamesMetavars, quat_system>>;
};

void test_system_to_combined_names_tag() {
  using system_to_combined_tag = control_system::Tags::SystemToCombinedNames;

  const std::unordered_map<std::string, std::string> system_to_combined_names =
      system_to_combined_tag::create_from_options<NamesMetavars>();

  CHECK(system_to_combined_names.count("LabelA") == 1);
  CHECK(system_to_combined_names.count("LabelB") == 1);
  CHECK(system_to_combined_names.count("Rotation") == 1);

  CHECK(system_to_combined_names.at("LabelA") == "LabelALabelB");
  CHECK(system_to_combined_names.at("LabelB") == "LabelALabelB");
  CHECK(system_to_combined_names.at("Rotation") == "Rotation");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ControlSystem.Tags", "[ControlSystem][Unit]") {
  test_all_tags();
  test_control_sys_inputs();
  test_individual_tags();
  test_system_to_combined_names_tag();
}
