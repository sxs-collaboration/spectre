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
#include "ControlSystem/Tags/IsActive.hpp"
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
struct Rotation {};
using system = control_system::TestHelpers::System<
    2, LabelA, control_system::TestHelpers::Measurement<LabelA>, 1>;
using system2 = control_system::TestHelpers::System<
    2, LabelA, control_system::TestHelpers::Measurement<LabelA>, 2>;
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

  using active_tag = control_system::Tags::IsActive<system>;
  TestHelpers::db::test_simple_tag<active_tag>("IsActive");
  // The create_from_options function doesn't actually use the metavars, only
  // the option_list type alias uses the metavars, so we pass an empty struct
  // here.
  const control_system::OptionHolder<system> active_holder{};
  const control_system::OptionHolder<system> inactive_holder{
      false, {}, {}, {}, {}};
  CHECK(active_tag::create_from_options<MetavarsEmpty>(active_holder));
  CHECK_FALSE(active_tag::create_from_options<MetavarsEmpty>(inactive_holder));
  CHECK_FALSE(active_tag::create_from_options<MetavarsEmpty>(
      active_holder, {"/fake/path"}, {{"FakeSpecName", "LabelA"}}));
  CHECK_FALSE(active_tag::create_from_options<MetavarsEmpty>(
      inactive_holder, {"/fake/path"}, {{"FakeSpecName", "LabelA"}}));
  CHECK(active_tag::create_from_options<MetavarsEmpty>(active_holder,
                                                       {"/fake/path"}, {}));
  CHECK_FALSE(active_tag::create_from_options<MetavarsEmpty>(
      inactive_holder, {"/fake/path"}, {}));
  CHECK(active_tag::create_from_options<MetavarsEmpty>(
      active_holder, std::nullopt, {{"FakeSpecName", "LabelA"}}));
  CHECK_FALSE(active_tag::create_from_options<MetavarsEmpty>(
      inactive_holder, std::nullopt, {{"FakeSpecName", "LabelA"}}));

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
}

void test_control_sys_inputs() {
  INFO("Test control system inputs");
  const double decrease_timescale_threshold = 1.0e-2;
  const double increase_timescale_threshold = 1.0e-4;
  const double increase_factor = 1.01;
  const double decrease_factor = 0.99;
  const double max_timescale = 10.0;
  const double min_timescale = 1.0e-3;
  const TimescaleTuner expected_tuner(
      std::vector<double>{1.}, max_timescale, min_timescale,
      decrease_timescale_threshold, increase_timescale_threshold,
      increase_factor, decrease_factor);
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
    return TimescaleTuner{std::vector<double>(num_components, 1.0),
                          max_timescale,
                          min_timescale,
                          decrease_timescale_threshold,
                          increase_timescale_threshold,
                          increase_factor,
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
      std::make_unique<FakeCreator>(
          std::unordered_map<std::string, size_t>{{system::name(), 2},
                                                  {quat_system::name(), 3}},
          2);
  const std::unique_ptr<DomainCreator<3>> creator_empty =
      std::make_unique<FakeCreator>(std::unordered_map<std::string, size_t>{},
                                    1);

  const TimescaleTuner created_tuner =
      tuner_tag::create_from_options<MetavarsEmpty>(holder, creator, 0.0);
  const TimescaleTuner quat_created_tuner =
      quat_tuner_tag::create_from_options<MetavarsEmpty>(quat_holder, creator,
                                                         0.0);
  const TimescaleTuner inactive_created_tuner =
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
      control_error_tag::create_from_options<MetavarsEmpty>(holder, creator);
  [[maybe_unused]] const control_error_tag2::type created_control_error2 =
      control_error_tag2::create_from_options<MetavarsEmpty>(holder2, creator);

  const std::unique_ptr<DomainCreator<3>> creator_error_0 =
      std::make_unique<FakeCreator>(std::unordered_map<std::string, size_t>{},
                                    0);

  CHECK_THROWS_WITH(
      control_error_tag::create_from_options<MetavarsEmpty>(holder,
                                                            creator_error_0),
      Catch::Contains(
          "ExcisionSphereA' or 'ExcisionSphereB' or 'ExcisionSphere"));
  CHECK_THROWS_WITH(control_error_tag2::create_from_options<MetavarsEmpty>(
                        holder2, creator_empty),
                    Catch::Contains("'ExcisionSphereB'"));

  using controller_tag = control_system::Tags::Controller<system>;

  Controller<2> expected_controller(0.3);
  expected_controller.set_initial_update_time(0.0);
  expected_controller.assign_time_between_updates(
      min(created_tuner.current_timescale()));

  const Controller<2> created_controller =
      controller_tag::create_from_options<MetavarsEmpty>(holder, creator, 0.0);

  CHECK(created_controller == expected_controller);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ControlSystem.Tags", "[ControlSystem][Unit]") {
  test_all_tags();
  test_control_sys_inputs();
  test_individual_tags();
}
