// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTriggers/Or.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/EventsAndDenseTriggers/DenseTriggers/TestTrigger.hpp"
#include "Options/Options.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using DenseTriggerBase = DenseTrigger<
    tmpl::list<DenseTriggers::Registrars::Or,
               TestHelpers::DenseTriggers::Registrars::TestTrigger>>;

void check(const bool time_runs_forward, const bool expected_is_ready,
           const bool expected_is_triggered, const double expected_next_check,
           const std::string& creation_string) noexcept {
  CAPTURE(creation_string);
  const auto box = db::create<db::AddSimpleTags<Tags::TimeStepId>>(
      TimeStepId(time_runs_forward, 0, Slab(0.0, 1.0).start()));
  const auto trigger = serialize_and_deserialize(
      TestHelpers::test_creation<std::unique_ptr<DenseTriggerBase>>(
          creation_string));

  CHECK(trigger->is_ready(box) == expected_is_ready);
  if (not expected_is_ready) {
    return;
  }
  const auto result = trigger->is_triggered(box);
  CHECK(result.is_triggered == expected_is_triggered);
  CHECK(result.next_check == expected_next_check);
}

void check_permutations(
    const bool expected_is_ready,
    const bool expected_is_triggered,
    const double expected_next_check,
    std::vector<std::pair<bool, bool>> is_ready_and_is_triggered,
    std::vector<double> next_checks) noexcept {
  alg::sort(is_ready_and_is_triggered);
  alg::sort(next_checks);
  do {
    do {
      for (const bool time_runs_forward : {true, false}) {
        const double sign = time_runs_forward ? 1.0 : -1.0;
        std::stringstream creation;
        creation << std::boolalpha;
        creation << "Or:\n";
        for (size_t i = 0; i < is_ready_and_is_triggered.size(); ++i) {
          creation << "  - TestTrigger:\n"
                   << "      IsReady: " << is_ready_and_is_triggered[i].first
                   << "\n"
                   << "      IsTriggered: "
                   << is_ready_and_is_triggered[i].second << "\n"
                   << "      NextCheck: " << sign * next_checks[i] << "\n";
        }
        check(time_runs_forward, expected_is_ready, expected_is_triggered,
              sign * expected_next_check, creation.str());
      }
    } while (cpp20::next_permutation(next_checks.begin(), next_checks.end()));
  } while (cpp20::next_permutation(is_ready_and_is_triggered.begin(),
                                   is_ready_and_is_triggered.end()));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.EventsAndDenseTriggers.DenseTriggers.Or",
                  "[Unit][Evolution]") {
  Parallel::register_derived_classes_with_charm<DenseTriggerBase>();

  check_permutations(true, false, std::numeric_limits<double>::infinity(), {},
                     {});

  check_permutations(false, false, 0.0, {{false, false}}, {5.0});
  check_permutations(true, false, 5.0, {{true, false}}, {5.0});
  check_permutations(true, true, 5.0, {{true, true}}, {5.0});

  check_permutations(false, false, 0.0, {{false, false}, {false, false}},
                     {5.0, 10.0});
  check_permutations(false, false, 0.0, {{false, false}, {true, false}},
                     {5.0, 10.0});
  check_permutations(true, false, 5.0, {{true, false}, {true, false}},
                     {5.0, 10.0});
  check_permutations(true, true, 5.0, {{true, false}, {true, true}},
                     {5.0, 10.0});
  check_permutations(true, true, 5.0, {{true, true}, {true, true}},
                     {5.0, 10.0});
}
