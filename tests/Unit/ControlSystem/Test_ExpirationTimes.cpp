// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "ControlSystem/CalculateMeasurementTimescales.hpp"
#include "ControlSystem/ExpirationTimes.hpp"
#include "ControlSystem/Protocols/ControlSystem.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Helpers/ControlSystem/SystemHelpers.hpp"
#include "Helpers/ControlSystem/TestStructs.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
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

template <size_t Index>
using OptionHolder = control_system::OptionHolder<FakeControlSystem<Index>>;

void check_expiration_times(
    const std::unordered_map<std::string, double>& initial_expiration_times,
    const std::unordered_map<std::string, double>&
        expected_initial_expiration_times) {
  CHECK(expected_initial_expiration_times.size() ==
        initial_expiration_times.size());
  for (auto& [expected_name, expected_expiration_time] :
       expected_initial_expiration_times) {
    CHECK(initial_expiration_times.count(expected_name) == 1);
    CHECK(initial_expiration_times.at(expected_name) ==
          expected_expiration_time);
  }
}

void test_expiration_time_construction() {
  const double initial_time = 0.9;
  constexpr int measurements_per_update = 4;

  const double timescale = 2.0;
  const TimescaleTuner tuner1(std::vector<double>{timescale}, 10.0, 1.0e-3,
                              1.0e-2, 1.0e-4, 1.01, 0.99);
  TimescaleTuner tuner2(0.1, 10.0, 1.0e-3, 1.0e-2, 1.0e-4, 1.01, 0.99);
  tuner2.resize_timescales(2);
  const Averager<1> averager(0.25, true);
  const double update_fraction = 0.3;
  const Controller<2> controller(update_fraction);
  const control_system::TestHelpers::ControlError<1> control_error{};

  OptionHolder<1> option_holder1(true, averager, controller, tuner1,
                                 control_error);
  OptionHolder<2> option_holder2(false, averager, controller, tuner1,
                                 control_error);
  OptionHolder<3> option_holder3(true, averager, controller, tuner2,
                                 control_error);

  using FakeCreator = control_system::TestHelpers::FakeCreator;

  const std::unique_ptr<DomainCreator<3>> creator1 =
      std::make_unique<FakeCreator>(std::unordered_map<std::string, size_t>{});
  const std::unique_ptr<DomainCreator<3>> creator2 =
      std::make_unique<FakeCreator>(std::unordered_map<std::string, size_t>{
          {FakeControlSystem<1>::name(), 1}});
  const std::unique_ptr<DomainCreator<3>> creator3 =
      std::make_unique<FakeCreator>(std::unordered_map<std::string, size_t>{
          {FakeControlSystem<1>::name(), 1},
          {FakeControlSystem<2>::name(), 1},
          {FakeControlSystem<3>::name(), 2}});

  {
    INFO("No control systems");
    const auto initial_expiration_times =
        control_system::initial_expiration_times(
            initial_time, measurements_per_update, creator1);

    const std::unordered_map<std::string, double>
        expected_initial_expiration_times{};

    check_expiration_times(expected_initial_expiration_times,
                           initial_expiration_times);
  }
  {
    INFO("One control system");
    const auto initial_expiration_times =
        control_system::initial_expiration_times(
            initial_time, measurements_per_update, creator2, option_holder1);

    const std::unordered_map<std::string, double>
        expected_initial_expiration_times{
            {FakeControlSystem<1>::name(),
             // This is ok to use here because we test it below
             control_system::function_of_time_expiration_time(
                 initial_time, DataVector{0.0},
                 control_system::calculate_measurement_timescales(
                     controller, tuner1, measurements_per_update),
                 measurements_per_update)}};

    check_expiration_times(expected_initial_expiration_times,
                           initial_expiration_times);
  }
  {
    INFO("Three control system");
    const auto initial_expiration_times =
        control_system::initial_expiration_times(
            initial_time, measurements_per_update, creator3, option_holder1,
            option_holder2, option_holder3);

    const std::unordered_map<std::string, double>
        expected_initial_expiration_times{
            {FakeControlSystem<1>::name(),
             control_system::function_of_time_expiration_time(
                 initial_time, DataVector{0.0},
                 control_system::calculate_measurement_timescales(
                     controller, tuner1, measurements_per_update),
                 measurements_per_update)},
            {FakeControlSystem<2>::name(),
             std::numeric_limits<double>::infinity()},
            {FakeControlSystem<3>::name(),
             control_system::function_of_time_expiration_time(
                 initial_time, DataVector{0.0},
                 control_system::calculate_measurement_timescales(
                     controller, tuner2, measurements_per_update),
                 measurements_per_update)}};

    check_expiration_times(expected_initial_expiration_times,
                           initial_expiration_times);
  }
}

void test_fot_measurement_expr_time() {
  const DataVector old_measurement_timescales{2.2, 3.3, 1.1};
  const DataVector new_measurement_timescales{1.2, 2.3, 3.4};

  const double time = 0.6;
  const int measurements_per_update = 3;

  const double fot_expr_time = control_system::function_of_time_expiration_time(
      time, old_measurement_timescales, new_measurement_timescales,
      measurements_per_update);
  const double expected_fot_expr_time =
      time + min(old_measurement_timescales) +
      measurements_per_update * min(new_measurement_timescales);

  CHECK(fot_expr_time == expected_fot_expr_time);

  const double measurement_expr_time =
      control_system::measurement_expiration_time(
          time, old_measurement_timescales, new_measurement_timescales,
          measurements_per_update);
  const double expected_measurement_expr_time =
      time + min(old_measurement_timescales) +
      (double(measurements_per_update) - 0.5) * min(new_measurement_timescales);

  CHECK(measurement_expr_time == expected_measurement_expr_time);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ControlSystem.ExpirationTimes",
                  "[ControlSystem][Unit]") {
  test_expiration_time_construction();
  test_fot_measurement_expr_time();
}
