// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <pup.h>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepChoosers/StepToTimes.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<StepChooser<StepChooserUse::Slab>,
                             tmpl::list<StepChoosers::StepToTimes>>,
                  tmpl::pair<TimeSequence<double>,
                             TimeSequences::all_time_sequences<double>>>;
  };
  using component_list = tmpl::list<>;
};

void check_case(const double now, std::vector<double> times,
                const std::optional<double>& expected_end) {
  const auto impl = [](const double local_now, const double last_step,
                       const std::vector<double>& impl_times) {
    CAPTURE(local_now);
    CAPTURE(impl_times);

    using Specified = TimeSequences::Specified<double>;
    const StepChoosers::StepToTimes step_to_times(
        std::make_unique<Specified>(impl_times));
    const std::unique_ptr<StepChooser<StepChooserUse::Slab>>
        step_to_times_base = std::make_unique<StepChoosers::StepToTimes>(
            std::make_unique<Specified>(impl_times));

    auto box = db::create<db::AddSimpleTags<
        Parallel::Tags::MetavariablesImpl<Metavariables>, Tags::Time>>(
        Metavariables{}, local_now);

    const auto answer = step_to_times(local_now, last_step);
    CHECK(answer.second);

    const auto& request = answer.first;
    CHECK(request.end == request.end_hard_limit);
    REQUIRE(request.size.has_value() == request.end.has_value());
    CHECK(not request.size_goal.has_value());
    CHECK(not request.size_hard_limit.has_value());

    CHECK(step_to_times_base->desired_step(last_step, box) == answer);
    CHECK(serialize_and_deserialize(step_to_times)(local_now, last_step) ==
          answer);
    CHECK(serialize_and_deserialize(step_to_times_base)
              ->desired_step(last_step, box) == answer);
    return request.end.has_value()
               ? std::optional(std::pair(*request.end, *request.size))
               : std::nullopt;
  };
  const auto result = impl(now, 1.0, times);
  alg::for_each(times, [](double& x) { return x = -x; });
  const auto backwards_result = impl(-now, -1.0, times);
  CHECK(backwards_result.has_value() == result.has_value());
  if (result.has_value()) {
    CHECK(backwards_result ==
          std::optional(std::pair(-result->first, -result->second)));
  }

  CHECK(result.has_value() == expected_end.has_value());
  if (expected_end.has_value()) {
    CHECK(result == std::optional(std::pair(
                        *expected_end, 2.0 / 3.0 * (*expected_end - now))));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.StepChoosers.StepToTimes", "[Unit][Time]") {
  register_factory_classes_with_charm<Metavariables>();

  check_case(3.0, {}, {});
  check_case(3.0, {1.0}, {});
  check_case(3.0, {3.0}, {});
  check_case(3.0, {5.0}, {5.0});

  const auto one_five_check = [](const std::vector<double>& times) {
    check_case(0.0, times, {1.0});
    check_case(1.0, times, {5.0});
    check_case(2.0, times, {5.0});
    check_case(5.0, times, {});
    check_case(7.0, times, {});
  };

  one_five_check({1.0, 5.0});
  one_five_check({5.0, 1.0});
  one_five_check({1.0, 5.0, 1.0, 5.0, 1.0, 5.0});

  const auto check_rounding = [](const double start, const double step) {
    CAPTURE(start);
    CAPTURE(step);
    static constexpr double infinity = std::numeric_limits<double>::infinity();
    check_case(std::nextafter(start, -infinity), {start, start + step},
               {start});
    check_case(start, {start, start + step}, {start + step});
    check_case(std::nextafter(start, +infinity), {start, start + step},
               {start + step});
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

  TestHelpers::test_creation<std::unique_ptr<StepChooser<StepChooserUse::Slab>>,
                             Metavariables>(
      "StepToTimes:\n"
      "  Times:\n"
      "    Specified:\n"
      "      Values: [5.0, 3.0, 6.0]");

  CHECK(not StepChoosers::StepToTimes{}.uses_local_data());
}
