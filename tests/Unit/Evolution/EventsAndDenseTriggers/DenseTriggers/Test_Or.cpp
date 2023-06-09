// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <initializer_list>
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
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Metavariables {
  using component_list = tmpl::list<>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<
        DenseTrigger, tmpl::list<DenseTriggers::Or,
                                 TestHelpers::DenseTriggers::TestTrigger>>>;
  };
};

void check(const bool time_runs_forward,
           const std::optional<bool>& expected_is_triggered,
           const std::optional<double>& expected_next_check_time,
           const std::string& creation_string) {
  CAPTURE(creation_string);
  const auto box = db::create<
      db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<Metavariables>,
                        Tags::TimeStepId, Tags::Time>>(
      Metavariables{}, TimeStepId(time_runs_forward, 0, Slab(0.0, 1.0).start()),
      0.0);
  Parallel::GlobalCache<Metavariables> cache{};
  const int array_index = 0;
  const void* component = nullptr;
  const auto trigger = serialize_and_deserialize(
      TestHelpers::test_creation<std::unique_ptr<DenseTrigger>, Metavariables>(
          creation_string));

  CHECK(trigger->is_triggered(box, cache, array_index, component) ==
        expected_is_triggered);
  CHECK(trigger->next_check_time(box, cache, array_index, component) ==
        expected_next_check_time);
}

void check_permutations(const std::optional<bool>& expected_is_triggered,
                        const std::optional<double>& expected_next_check,
                        std::vector<std::optional<bool>> is_triggered,
                        std::vector<std::optional<double>> next_checks) {
  alg::sort(is_triggered);
  alg::sort(next_checks);
  do {
    do {
      for (const bool time_runs_forward : {true, false}) {
        const double sign = time_runs_forward ? 1.0 : -1.0;
        const auto set_sign =
            [&sign](const std::optional<double>& arg) -> std::optional<double> {
          if (arg.has_value()) {
            return std::optional{sign * *arg};
          } else {
            return std::nullopt;
          }
        };
        using NotReady = TestHelpers::DenseTriggers::TestTrigger::NotReady;
        std::stringstream creation;
        creation << std::boolalpha;
        creation << "Or:\n";
        for (size_t i = 0; i < is_triggered.size(); ++i) {
          creation << "  - TestTrigger:\n"
                   << "      IsTriggered: " << NotReady::format(is_triggered[i])
                   << "\n"
                   << "      NextCheck: "
                   << NotReady::format(set_sign(next_checks[i])) << "\n";
        }
        check(time_runs_forward, expected_is_triggered,
              set_sign(expected_next_check), creation.str());
      }
    } while (cpp20::next_permutation(next_checks.begin(), next_checks.end()));
  } while (cpp20::next_permutation(is_triggered.begin(), is_triggered.end()));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.EventsAndDenseTriggers.DenseTriggers.Or",
                  "[Unit][Evolution]") {
  register_factory_classes_with_charm<Metavariables>();

  check_permutations(false, std::numeric_limits<double>::infinity(), {}, {});

  {
    const std::initializer_list<
        std::pair<std::optional<bool>, std::vector<std::optional<bool>>>>
        is_triggered_tests{
            {std::nullopt, {std::nullopt}}, {false, {false}}, {true, {true}}};
    const std::initializer_list<
        std::pair<std::optional<double>, std::vector<std::optional<double>>>>
        next_check_time_tests{{std::nullopt, {std::nullopt}}, {5.0, {5.0}}};
    for (const auto& is_triggered_test : is_triggered_tests) {
      for (const auto& next_check_time_test : next_check_time_tests) {
        check_permutations(is_triggered_test.first, next_check_time_test.first,
                           is_triggered_test.second,
                           next_check_time_test.second);
      }
    }
  }

  {
    const std::initializer_list<
        std::pair<std::optional<bool>, std::vector<std::optional<bool>>>>
        is_triggered_tests{{std::nullopt, {std::nullopt, std::nullopt}},
                           {std::nullopt, {std::nullopt, false}},
                           {true, {std::nullopt, true}},
                           {false, {false, false}},
                           {true, {false, true}},
                           {true, {true, true}}};
    const std::initializer_list<
        std::pair<std::optional<double>, std::vector<std::optional<double>>>>
        next_check_time_tests{{std::nullopt, {std::nullopt, std::nullopt}},
                              {std::nullopt, {std::nullopt, 5.0}},
                              {5.0, {5.0, 10.0}}};
    for (const auto& is_triggered_test : is_triggered_tests) {
      for (const auto& next_check_time_test : next_check_time_tests) {
        check_permutations(is_triggered_test.first, next_check_time_test.first,
                           is_triggered_test.second,
                           next_check_time_test.second);
      }
    }
  }
}
