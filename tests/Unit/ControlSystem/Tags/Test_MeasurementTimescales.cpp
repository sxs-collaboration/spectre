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

namespace OptionTags {
struct InitialTime;
struct InitialTimeStep;
}  // namespace OptionTags

namespace {
template <size_t Index>
struct Label {};
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
  using measurement =
      control_system::TestHelpers::Measurement<Label<Index % 3>>;
  using simple_tags = tmpl::list<>;
  using control_error = control_system::TestHelpers::ControlError;
  struct process_measurement {
    template <typename Submeasurement>
    using argument_tags = tmpl::list<>;
  };
};

struct Metavariables {
  static constexpr size_t volume_dim = 3;
  // This will create 5 control systems and 3 measurements. Two measurements
  // have two control systems and one measurement has only one control system.
  // The indices are chosen because of the % 3 above
  using control_systems = tmpl::list<FakeControlSystem<1>, FakeControlSystem<2>,
                                     FakeControlSystem<4>, FakeControlSystem<5>,
                                     FakeControlSystem<6>>;
  using component_list =
      control_system::control_components<Metavariables, control_systems>;
};

template <size_t DerivOrder>
void test_calculate_measurement_timescales() {
  INFO("Test calculate measurement timescales");
  const double timescale = 20.0;
  const TimescaleTuner<true> tuner(std::vector<double>{timescale}, 10.0, 1.0e-3,
                                   1.0e-4, 1.01, 1.0e-2, 0.99);
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
      tmpl::size<measurement_tag::option_tags<Metavariables>>::value == 9);

  using FakeCreator = control_system::TestHelpers::FakeCreator;

  const double time_step = 0.2;
  {
    const double averaging_fraction = 0.25;
    const Averager<1> averager(averaging_fraction, true);
    const double update_fraction = 0.3;
    const Controller<2> controller(update_fraction);
    const control_system::TestHelpers::ControlError control_error{};

    const double timescale_long = 27.0;
    const TimescaleTuner<true> tuner1(
        std::vector<double>{timescale_long, timescale_long * 2.0}, 10.0, 1.0e-3,
        1.0e-4, 1.01, 1.0e-2, 0.99);
    const double timescale_short = 0.5;
    TimescaleTuner<true> tuner2(timescale_short, 10.0, 1.0e-3, 1.0e-4, 1.01,
                                1.0e-2, 0.99);
    tuner2.resize_timescales(2);
    const TimescaleTuner<true> tuner4 = tuner2;
    const TimescaleTuner<true>& tuner5 = tuner1;
    const TimescaleTuner<true>& tuner6 = tuner1;

    OptionHolder<1> option_holder1(true, averager, controller, tuner1,
                                   control_error);
    // Control system 2 is not active so the measurement timescale and
    // expiration time should both be infinity
    OptionHolder<2> option_holder2(false, averager, controller, tuner2,
                                   control_error);
    OptionHolder<4> option_holder4(true, averager, controller, tuner4,
                                   control_error);
    OptionHolder<5> option_holder5(true, averager, controller, tuner5,
                                   control_error);
    OptionHolder<6> option_holder6(false, averager, controller, tuner6,
                                   control_error);

    const std::unique_ptr<DomainCreator<3>> creator =
        std::make_unique<FakeCreator>(std::unordered_map<std::string, size_t>{
            {FakeControlSystem<1>::name(), 2},
            {FakeControlSystem<2>::name(), 2},
            {FakeControlSystem<4>::name(), 2},
            {FakeControlSystem<5>::name(), 2},
            {FakeControlSystem<6>::name(), 2}});

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
                           FakeControlSystem<4>>,
                       control_system::OptionTags::ControlSystemInputs<
                           FakeControlSystem<5>>,
                       control_system::OptionTags::ControlSystemInputs<
                           FakeControlSystem<6>>>>);
    const int measurements_per_update = 4;
    const measurement_tag::type timescales =
        measurement_tag::create_from_options<Metavariables>(
            measurements_per_update, creator, initial_time, time_step,
            option_holder1, option_holder2, option_holder4, option_holder5,
            option_holder6);
    CHECK(timescales.size() == 3);
    CHECK(timescales.count("Controlled1Controlled4") == 1);
    CHECK(timescales.count("Controlled2Controlled5") == 1);
    CHECK(timescales.count("Controlled6") == 1);

    const double measure_time14 =
        std::min(min(control_system::calculate_measurement_timescales(
                     controller, tuner1, measurements_per_update)),
                 min(control_system::calculate_measurement_timescales(
                     controller, tuner4, measurements_per_update)));
    // Control system 2 is turned off
    const double measure_time25 =
        min(control_system::calculate_measurement_timescales(
            controller, tuner5, measurements_per_update));
    const double measure_time6 = std::numeric_limits<double>::infinity();
    const double expr_time14 =
        initial_time + update_fraction * timescale_short - 0.5 * measure_time14;
    const double expr_time25 =
        initial_time + update_fraction * timescale_long - 0.5 * measure_time25;
    const double expr_time6 = std::numeric_limits<double>::infinity();

    CHECK(timescales.at("Controlled1Controlled4")->time_bounds() ==
          std::array{initial_time, expr_time14});
    CHECK(timescales.at("Controlled1Controlled4")->func(1.0)[0] ==
          DataVector{measure_time14});

    CHECK(timescales.at("Controlled2Controlled5")->time_bounds() ==
          std::array{initial_time, expr_time25});
    CHECK(timescales.at("Controlled2Controlled5")->func(2.0)[0] ==
          DataVector{measure_time25});

    CHECK(timescales.at("Controlled6")->time_bounds() ==
          std::array{initial_time, expr_time6});
    CHECK(timescales.at("Controlled6")->func(2.0)[0] ==
          DataVector{measure_time6});
  }

  CHECK_THROWS_WITH(
      ([]() {
        const TimescaleTuner<true> tuner1(std::vector<double>{27.0}, 10.0,
                                          1.0e-3, 1.0e-4, 1.01, 1.0e-2, 0.99);
        const TimescaleTuner<true> tuner2(std::vector<double>{0.1}, 10.0,
                                          1.0e-3, 1.0e-4, 1.01, 1.0e-2, 0.99);
        const Averager<1> averager(0.25, true);
        const Controller<2> controller(0.3);
        const control_system::TestHelpers::ControlError control_error{};

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
      Catch::Matchers::ContainsSubstring(
          "Control systems can only be used in forward-in-time evolutions."));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ControlSystem.Tags.MeasurementTimescales",
                  "[ControlSystem][Unit]") {
  test_calculate_measurement_timescales<2>();
  test_calculate_measurement_timescales<3>();
  test_measurement_tag();
}
