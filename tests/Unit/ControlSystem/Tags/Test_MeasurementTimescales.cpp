// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "ControlSystem/Averager.hpp"
#include "ControlSystem/CalculateMeasurementTimescales.hpp"
#include "ControlSystem/Component.hpp"
#include "ControlSystem/Controller.hpp"
#include "ControlSystem/Protocols/ControlSystem.hpp"
#include "ControlSystem/Tags/MeasurementTimescales.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Helpers/ControlSystem/SystemHelpers.hpp"
#include "Helpers/ControlSystem/TestStructs.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
const double initial_time = 0.9;

template <size_t Index>
struct FakeControlSystem
    : tt::ConformsTo<control_system::protocols::ControlSystem> {
  static constexpr size_t deriv_order = 2;
  static std::string name() { return "Controlled"s + get_output(Index); }
  static std::optional<std::string> component_name(
      const size_t i, const size_t /*num_components*/) {
    return get_output(i);
  }
  using measurement = control_system::TestHelpers::Measurement<
      control_system::TestHelpers::TestStructs_detail::LabelA>;
  using simple_tags = tmpl::list<>;
  using control_error = control_system::TestHelpers::ControlError<1>;
  struct process_measurement {
    using argument_tags = tmpl::list<>;
  };
};

struct Metavariables {
  static constexpr size_t volume_dim = 3;
  using control_systems = tmpl::list<FakeControlSystem<1>, FakeControlSystem<2>,
                                     FakeControlSystem<3>>;
  using component_list =
      control_system::control_components<Metavariables, control_systems>;
};

struct MetavariablesReplace : Metavariables {
  static constexpr bool override_functions_of_time = true;
};

struct MetavariablesNoControlSystems {
  static constexpr size_t volume_dim = 3;
  using component_list = tmpl::list<>;
};

template <size_t DerivOrder>
void test_calculate_measurement_timescales() {
  INFO("Test calculate measurement timescales");
  const double timescale = 20.0;
  const TimescaleTuner tuner(std::vector<double>{timescale}, 10.0, 1.0e-3,
                             1.0e-2, 1.0e-4, 1.01, 0.99);
  const double update_fraction = 0.25;
  const Controller<DerivOrder> controller(update_fraction);

  const int measurements_per_update = 4;
  const auto measurement_timescales =
      control_system::calculate_measurement_timescales(controller, tuner,
                                                       measurements_per_update);

  const DataVector expected_measurement_timescales =
      DataVector{{timescale}} * update_fraction /
      double(measurements_per_update);

  CHECK(expected_measurement_timescales == measurement_timescales);
}

template <size_t Index>
using OptionHolder = control_system::OptionHolder<FakeControlSystem<Index>>;

void test_measurement_tag() {
  INFO("Test measurement tag");
  using measurement_tag = control_system::Tags::MeasurementTimescales;
  static_assert(
      tmpl::size<
          measurement_tag::option_tags<MetavariablesNoControlSystems>>::value ==
      2);
  static_assert(
      tmpl::size<measurement_tag::option_tags<Metavariables>>::value == 7);

  using FakeCreator = control_system::TestHelpers::FakeCreator;

  const double time_step = 0.2;
  {
    const double timescale1 = 27.0;
    const TimescaleTuner tuner1(std::vector<double>{timescale1}, 10.0, 1.0e-3,
                                1.0e-2, 1.0e-4, 1.01, 0.99);
    const double timescale2 = 0.5;
    TimescaleTuner tuner2(timescale2, 10.0, 1.0e-3, 1.0e-2, 1.0e-4, 1.01, 0.99);
    tuner2.resize_timescales(2);
    const double averaging_fraction = 0.25;
    const Averager<1> averager(averaging_fraction, true);
    const double update_fraction = 0.3;
    const Controller<2> controller(update_fraction);
    const control_system::TestHelpers::ControlError<1> control_error{};

    OptionHolder<1> option_holder1(true, averager, controller, tuner1,
                                   control_error);
    // Control system 2 is not active so the measurement timescale and
    // expiration time should both be infinity
    OptionHolder<2> option_holder2(false, averager, controller, tuner1,
                                   control_error);
    OptionHolder<3> option_holder3(true, averager, controller, tuner2,
                                   control_error);

    const std::unique_ptr<DomainCreator<3>> creator =
        std::make_unique<FakeCreator>(std::unordered_map<std::string, size_t>{
            {FakeControlSystem<1>::name(), 1},
            {FakeControlSystem<2>::name(), 1},
            {FakeControlSystem<3>::name(), 2}});

    static_assert(
        std::is_same_v<
            measurement_tag::option_tags<Metavariables>,
            tmpl::list<control_system::OptionTags::MeasurementsPerUpdate,
                       domain::OptionTags::DomainCreator<3>,
                       ::OptionTags::InitialTime, ::OptionTags::InitialTimeStep,
                       control_system::OptionTags::ControlSystemInputs<
                           FakeControlSystem<1>>,
                       control_system::OptionTags::ControlSystemInputs<
                           FakeControlSystem<2>>,
                       control_system::OptionTags::ControlSystemInputs<
                           FakeControlSystem<3>>>>);
    const int measurements_per_update = 4;
    const measurement_tag::type timescales =
        measurement_tag::create_from_options<Metavariables>(
            measurements_per_update, creator, initial_time, time_step,
            option_holder1, option_holder2, option_holder3);
    CHECK(timescales.size() == 3);

    const double measure_time1 =
        control_system::calculate_measurement_timescales(
            controller, tuner1, measurements_per_update)[0];
    const double measure_time2 =
        control_system::calculate_measurement_timescales(
            controller, tuner2, measurements_per_update)[0];
    const double expr_time1 =
        initial_time + update_fraction * timescale1 - 0.5 * measure_time1;
    const double expr_time2 =
        initial_time + update_fraction * timescale2 - 0.5 * measure_time2;
    CHECK(timescales.at("Controlled1")->time_bounds() ==
          std::array{initial_time, expr_time1});
    CHECK(timescales.at("Controlled1")->func(2.0)[0] ==
          DataVector{measure_time1});
    CHECK(timescales.at("Controlled1")->func(3.0)[0] ==
          DataVector{measure_time1});
    CHECK(timescales.at("Controlled2")->time_bounds() ==
          std::array{initial_time, std::numeric_limits<double>::infinity()});
    CHECK(timescales.at("Controlled2")->func(2.0)[0] ==
          DataVector{std::numeric_limits<double>::infinity()});
    CHECK(timescales.at("Controlled2")->func(3.0)[0] ==
          DataVector{std::numeric_limits<double>::infinity()});
    CHECK(timescales.at("Controlled3")->time_bounds() ==
          std::array{initial_time, expr_time2});
    CHECK(timescales.at("Controlled3")->func(1.0)[0] ==
          DataVector{measure_time2, measure_time2});

    // Replace Controlled2 with something read in from an h5 file. This means
    // the measurement timescale and expiration time for Controlled2 is
    // infinity still, but for a different reason.
    static_assert(
        std::is_same_v<
            measurement_tag::option_tags<MetavariablesReplace>,
            tmpl::list<
                control_system::OptionTags::MeasurementsPerUpdate,
                domain::OptionTags::DomainCreator<3>,
                domain::FunctionsOfTime::OptionTags::FunctionOfTimeFile,
                domain::FunctionsOfTime::OptionTags::FunctionOfTimeNameMap,
                ::OptionTags::InitialTime, ::OptionTags::InitialTimeStep,
                control_system::OptionTags::ControlSystemInputs<
                    FakeControlSystem<1>>,
                control_system::OptionTags::ControlSystemInputs<
                    FakeControlSystem<2>>,
                control_system::OptionTags::ControlSystemInputs<
                    FakeControlSystem<3>>>>);
    const auto replaced_timescales =
        measurement_tag::create_from_options<MetavariablesReplace>(
            4, creator, {"FakeFileName"}, {{"FakeSpecName", "Controlled2"}},
            initial_time, time_step, option_holder1, option_holder2,
            option_holder3);
    CHECK(replaced_timescales.at("Controlled2")->time_bounds() ==
          std::array{initial_time, std::numeric_limits<double>::infinity()});
    CHECK(replaced_timescales.at("Controlled2")->func(2.0)[0][0] ==
          std::numeric_limits<double>::infinity());
  }

  CHECK_THROWS_WITH(
      ([]() {
        const TimescaleTuner tuner1(std::vector<double>{27.0}, 10.0, 1.0e-3,
                                    1.0e-2, 1.0e-4, 1.01, 0.99);
        const TimescaleTuner tuner2(std::vector<double>{0.1}, 10.0, 1.0e-3,
                                    1.0e-2, 1.0e-4, 1.01, 0.99);
        const Averager<1> averager(0.25, true);
        const Controller<2> controller(0.3);
        const control_system::TestHelpers::ControlError<1> control_error{};

        OptionHolder<1> option_holder1(true, averager, controller, tuner1,
                                       control_error);
        OptionHolder<2> option_holder2(true, averager, controller, tuner1,
                                       control_error);
        OptionHolder<3> option_holder3(true, averager, controller, tuner2,
                                       control_error);

        const std::unique_ptr<DomainCreator<3>> creator =
            std::make_unique<FakeCreator>(
                std::unordered_map<std::string, size_t>{
                    {FakeControlSystem<1>::name(), 1},
                    {FakeControlSystem<2>::name(), 1},
                    {FakeControlSystem<3>::name(), 1}});

        measurement_tag::create_from_options<Metavariables>(
            4, creator, initial_time, -1.0, option_holder1, option_holder2,

            option_holder3);
      })(),
      Catch::Contains(
          "Control systems can only be used in forward-in-time evolutions."));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ControlSystem.Tags.MeasurementTimescales",
                  "[ControlSystem][Unit]") {
  test_calculate_measurement_timescales<2>();
  test_calculate_measurement_timescales<3>();
  test_measurement_tag();
}
