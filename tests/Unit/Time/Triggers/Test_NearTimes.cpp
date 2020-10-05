// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <pup.h>
#include <pup_stl.h>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/Triggers/NearTimes.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/TMPL.hpp"

SPECTRE_TEST_CASE("Unit.Time.Triggers.NearTimes", "[Unit][Time]") {
  using TriggerType = Trigger<tmpl::list<Triggers::Registrars::NearTimes>>;
  Parallel::register_derived_classes_with_charm<TriggerType>();
  Parallel::register_derived_classes_with_charm<TimeSequence<double>>();

  using Direction = Triggers::NearTimes<>::Direction;
  using Unit = Triggers::NearTimes<>::Unit;

  const auto check =
      [](const double time_in, const double range_in,
         const std::vector<double>& trigger_times_in, const Direction direction,
         const bool expected) noexcept {
    const auto check_signs =
        [&direction, &expected, &trigger_times_in, &time_in](
            const TimeDelta time_step_in, const double range,
            const Unit unit) noexcept {
      const auto check_calls = [&direction, &expected, &range, &unit](
          std::vector<double> trigger_times, const double time,
          const TimeDelta& time_step) noexcept {
        CAPTURE_PRECISE(trigger_times);
        CAPTURE_PRECISE(range);
        CAPTURE_PRECISE(static_cast<int>(unit));
        CAPTURE_PRECISE(static_cast<int>(direction));
        CAPTURE_PRECISE(time);
        CAPTURE_PRECISE(time_step);
        const std::unique_ptr<TriggerType> trigger =
            std::make_unique<Triggers::NearTimes<>>(
                std::make_unique<TimeSequences::Specified<double>>(
                    std::move(trigger_times)),
                range, unit, direction);
        const auto sent_trigger = serialize_and_deserialize(trigger);

        const auto box =
            db::create<db::AddSimpleTags<Tags::Time, Tags::TimeStep>>(
                time, time_step);

        CHECK(trigger->is_triggered(box) == expected);
        CHECK(sent_trigger->is_triggered(box) == expected);
      };

      check_calls(trigger_times_in, time_in, time_step_in);
      std::vector<double> negated_trigger_times = trigger_times_in;
      alg::for_each(negated_trigger_times, [](double& x) noexcept { x = -x; });
      check_calls(negated_trigger_times, -time_in, -time_step_in);
    };

    // Absolute position of slab should not matter.
    // Slab size 100, step size 10
    const TimeDelta large_step = Slab(10000.0, 10100.0).duration() / 10;
    // Slab size 0.1, step_size 0.1
    const TimeDelta small_step = Slab(10000.0, 10000.1).duration();

    check_signs(large_step, range_in, Unit::Time);
    check_signs(large_step, range_in / 10.0, Unit::Step);
    check_signs(large_step, range_in / 100.0, Unit::Slab);
    check_signs(small_step, range_in, Unit::Time);
    check_signs(small_step, range_in / 0.1, Unit::Step);
    check_signs(small_step, range_in / 0.1, Unit::Slab);
  };

  check(5.0, 3.0, {}, Direction::Both, false);
  check(5.0, 3.0, {}, Direction::Before, false);
  check(5.0, 3.0, {}, Direction::After, false);
  check(5.0, 3.0, {6.0}, Direction::Both, true);
  check(5.0, 3.0, {6.0}, Direction::Before, true);
  check(5.0, 3.0, {6.0}, Direction::After, false);
  check(5.0, 3.0, {4.0}, Direction::Both, true);
  check(5.0, 3.0, {4.0}, Direction::Before, false);
  check(5.0, 3.0, {4.0}, Direction::After, true);
  check(5.0, 3.0, {4.0, 6.0}, Direction::Both, true);
  check(5.0, 3.0, {4.0, 6.0}, Direction::Before, true);
  check(5.0, 3.0, {4.0, 6.0}, Direction::After, true);
  check(5.0, 3.0, {9.0}, Direction::Both, false);
  check(5.0, 3.0, {9.0}, Direction::Before, false);
  check(5.0, 3.0, {9.0}, Direction::After, false);
  check(5.0, 3.0, {1.0}, Direction::Both, false);
  check(5.0, 3.0, {1.0}, Direction::Before, false);
  check(5.0, 3.0, {1.0}, Direction::After, false);
  check(5.0, 3.0, {1.0, 9.0}, Direction::Both, false);
  check(5.0, 3.0, {1.0, 9.0}, Direction::Before, false);
  check(5.0, 3.0, {1.0, 9.0}, Direction::After, false);

  TestHelpers::test_factory_creation<TriggerType>(
      "NearTimes:\n"
      "  Times:\n"
      "    Specified:\n"
      "      Values: [2.0, 1.0, 3.0, 2.0]\n"
      "  Range: 0.3\n"
      "  Direction: Before\n"
      "  Unit: Time");
  CHECK(TestHelpers::test_creation<Unit>("Time") == Unit::Time);
  CHECK(TestHelpers::test_creation<Unit>("Step") == Unit::Step);
  CHECK(TestHelpers::test_creation<Unit>("Slab") == Unit::Slab);
  CHECK(TestHelpers::test_creation<Direction>("Before") == Direction::Before);
  CHECK(TestHelpers::test_creation<Direction>("After") == Direction::After);
  CHECK(TestHelpers::test_creation<Direction>("Both") == Direction::Both);
}
