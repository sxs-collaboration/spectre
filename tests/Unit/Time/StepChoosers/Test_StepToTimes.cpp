// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <pup.h>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepChoosers/StepToTimes.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Metavariables {
  using component_list = tmpl::list<>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.StepChoosers.StepToTimes", "[Unit][Time]") {
  using StepChooserType =
      StepChooser<tmpl::list<StepChoosers::Registrars::StepToTimes>>;
  using StepToTimes = StepChoosers::StepToTimes<>;

  Parallel::register_derived_classes_with_charm<StepChooserType>();
  Parallel::register_derived_classes_with_charm<TimeSequence<double>>();

  const auto requested = [](const double now, std::vector<double> times,
                            const double step) noexcept {
    double result = -1.0;
    const auto impl =
        [&result, &step](const TimeStepId& now_id,
                         const std::vector<double>& impl_times) noexcept {
      CAPTURE(now_id);
      CAPTURE(impl_times);
      CAPTURE(step);

      using Specified = TimeSequences::Specified<double>;
      const StepToTimes step_to_times(std::make_unique<Specified>(impl_times));
      const std::unique_ptr<StepChooserType> step_to_times_base =
          std::make_unique<StepToTimes>(
              std::make_unique<Specified>(impl_times));

      const Parallel::GlobalCache<Metavariables> cache{{}};
      const auto box = db::create<db::AddSimpleTags<Tags::TimeStepId>>(now_id);

      const double answer = step_to_times(now_id, step, cache);
      if (result == -1.0) {
        result = answer;
      } else {
        CHECK(result == answer);
      }
      CHECK(step_to_times_base->desired_step(step, box, cache) == result);
      CHECK(serialize_and_deserialize(step_to_times)(now_id, step, cache) ==
            result);
      CHECK(serialize_and_deserialize(step_to_times_base)
                ->desired_step(step, box, cache) == result);
    };
    impl(TimeStepId(true, 0, Slab(now, now + 1.0).start()), times);
    alg::for_each(times, [](double& x) noexcept { return x = -x; });
    impl(TimeStepId(false, 0, Slab(-now, -now + 1.0).start()), times);
    return result;
  };

  static constexpr double infinity = std::numeric_limits<double>::infinity();

  CHECK(requested(3.0, {}, infinity) == infinity);
  CHECK(requested(3.0, {1.0}, infinity) == infinity);
  CHECK(requested(3.0, {3.0}, infinity) == infinity);
  CHECK(requested(3.0, {5.0}, infinity) == 2.0);
  {
    const double request = requested(3.0, {5.0}, 1.0);
    CHECK(request > 1.0);
    CHECK(request < 2.0);
  }
  {
    const double request = requested(3.0, {5.0}, 1.9375);
    CHECK(request > 1.0);
    CHECK(request < 1.9375);
  }
  {
    const double request = requested(3.0, {5.0}, 1.0625);
    CHECK(request > 1.0);
    CHECK(request < 2.0);
  }

  const auto one_five_check = [&requested](
      const std::vector<double>& times) noexcept {
    CHECK(requested(0.0, times, infinity) == 1.0);
    CHECK(requested(1.0, times, infinity) == 4.0);
    CHECK(requested(2.0, times, infinity) == 3.0);
    CHECK(requested(5.0, times, infinity) == infinity);
    CHECK(requested(7.0, times, infinity) == infinity);
  };

  one_five_check({1.0, 5.0});
  one_five_check({5.0, 1.0});
  one_five_check({1.0, 5.0, 1.0, 5.0, 1.0, 5.0});

  const auto check_rounding = [&requested](const double start,
                                           const double step) noexcept {
    CAPTURE(start);
    CAPTURE(step);
    auto scaled_approx = Approx::custom().scale(std::abs(start));
    CHECK(requested(std::nextafter(start, +infinity), {start, start + step},
                    infinity) == scaled_approx(step));
    CHECK(requested(std::nextafter(start, -infinity), {start, start + step},
                    infinity) == scaled_approx(step));
  };

  check_rounding(0.0, 1.0);
  check_rounding(1.0, 1.0);
  check_rounding(-1.0, 1.0);
  check_rounding(1.0e5, 1.0);
  check_rounding(-1.0e5, 1.0);
  check_rounding(1.0, 1.0e5);
  check_rounding(-1.0, 1.0e5);
  check_rounding(1.0e5, 1.0e5);
  check_rounding(-1.0e5, 1.0e5);

  TestHelpers::test_factory_creation<StepChooserType>(
      "StepToTimes:\n"
      "  Times:\n"
      "    Specified:\n"
      "      Values: [5.0, 3.0, 6.0]");
}
