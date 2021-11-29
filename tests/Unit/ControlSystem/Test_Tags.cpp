// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "ControlSystem/Averager.hpp"
#include "ControlSystem/Component.hpp"
#include "ControlSystem/Controller.hpp"
#include "ControlSystem/Tags.hpp"
#include "ControlSystem/Tags/FunctionsOfTimeInitialize.hpp"
#include "ControlSystem/Tags/MeasurementTimescales.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/ControlSystem/TestStructs.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

namespace {
void test_all_tags() {
  INFO("Test all tags");
  using name_tag = control_system::Tags::ControlSystemName;
  TestHelpers::db::test_simple_tag<name_tag>("ControlSystemName");
  using averager_tag = control_system::Tags::Averager<2>;
  TestHelpers::db::test_simple_tag<averager_tag>("Averager");
  using timescaletuner_tag = control_system::Tags::TimescaleTuner;
  TestHelpers::db::test_simple_tag<timescaletuner_tag>("TimescaleTuner");
  using controller_tag = control_system::Tags::Controller<2>;
  TestHelpers::db::test_simple_tag<controller_tag>("Controller");
  using fot_tag = control_system::Tags::FunctionsOfTimeInitialize;
  TestHelpers::db::test_simple_tag<fot_tag>("FunctionsOfTime");

  using system = control_system::TestHelpers::System<
      2, control_system::TestHelpers::TestStructs_detail::LabelA,
      control_system::TestHelpers::Measurement<
          control_system::TestHelpers::TestStructs_detail::LabelA>>;
  using control_system_inputs_tag =
      control_system::Tags::ControlSystemInputs<system>;
  TestHelpers::db::test_simple_tag<control_system_inputs_tag>(
      "ControlSystemInputs");

  using measurement_tag = control_system::Tags::MeasurementTimescales;
  TestHelpers::db::test_simple_tag<measurement_tag>("MeasurementTimescales");
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
      {1.}, max_timescale, min_timescale, decrease_timescale_threshold,
      increase_timescale_threshold, increase_factor, decrease_factor);
  const Averager<2> expected_averager(0.25, true);
  const Controller<2> expected_controller(0.3);
  const std::string expected_name{"LabelA"};

  using system = control_system::TestHelpers::System<
      2, control_system::TestHelpers::TestStructs_detail::LabelA,
      control_system::TestHelpers::Measurement<
          control_system::TestHelpers::TestStructs_detail::LabelA>>;
  const auto input_holder = TestHelpers::test_option_tag<
      control_system::OptionTags::ControlSystemInputs<system>>(
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
      "  DecreaseFactor: 0.99\n");
  CHECK(expected_averager == input_holder.averager);
  CHECK(expected_controller == input_holder.controller);
  CHECK(expected_tuner == input_holder.tuner);
  CHECK(expected_name ==
        std::decay_t<decltype(input_holder)>::control_system::name());
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ControlSystem.Tags", "[ControlSystem][Unit]") {
  test_all_tags();
  test_control_sys_inputs();
}
