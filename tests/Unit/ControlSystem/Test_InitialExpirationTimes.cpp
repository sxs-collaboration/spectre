// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>
#include <unordered_map>

#include "ControlSystem/InitialExpirationTimes.hpp"
#include "ControlSystem/Protocols/ControlSystem.hpp"
#include "ControlSystem/Tags.hpp"
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
  using measurement = control_system::TestHelpers::Measurement<
      control_system::TestHelpers::TestStructs_detail::LabelA>;
  using simple_tags = tmpl::list<>;
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
}  // namespace

SPECTRE_TEST_CASE("Unit.ControlSystem.ConstructInitialExpirationTimes",
                  "[ControlSystem][Unit]") {
  const double initial_time = 1.3;
  const double initial_time_step = 0.1;

  const double timescale = 2.0;
  const TimescaleTuner tuner1({timescale}, 10.0, 1.0e-3, 1.0e-2, 1.0e-4, 1.01,
                              0.99);
  const TimescaleTuner tuner2({0.1}, 10.0, 1.0e-3, 1.0e-2, 1.0e-4, 1.01, 0.99);
  const Averager<2> averager(0.25, true);
  const double update_fraction = 0.3;
  const Controller<2> controller(update_fraction);

  OptionHolder<1> option_holder1(averager, controller, tuner1);
  OptionHolder<2> option_holder2(averager, controller, tuner1);
  OptionHolder<3> option_holder3(averager, controller, tuner2);

  {
    INFO("No control systems");
    const auto initial_expiration_times =
        control_system::initial_expiration_times(initial_time,
                                                 initial_time_step);

    const std::unordered_map<std::string, double>
        expected_initial_expiration_times{};

    check_expiration_times(expected_initial_expiration_times,
                           initial_expiration_times);
  }
  {
    INFO("One control system");
    const auto initial_expiration_times =
        control_system::initial_expiration_times(
            initial_time, initial_time_step, option_holder1);

    const std::unordered_map<std::string, double>
        expected_initial_expiration_times{
            {FakeControlSystem<1>::name(),
             initial_time + update_fraction * timescale}};

    check_expiration_times(expected_initial_expiration_times,
                           initial_expiration_times);
  }
  {
    INFO("Three control system");
    const auto initial_expiration_times =
        control_system::initial_expiration_times(
            initial_time, initial_time_step, option_holder1, option_holder2,
            option_holder3);

    const std::unordered_map<std::string, double>
        expected_initial_expiration_times{
            {FakeControlSystem<1>::name(),
             initial_time + update_fraction * timescale},
            {FakeControlSystem<2>::name(),
             initial_time + update_fraction * timescale},
            {FakeControlSystem<3>::name(), initial_time + initial_time_step}};

    check_expiration_times(expected_initial_expiration_times,
                           initial_expiration_times);
  }
}
