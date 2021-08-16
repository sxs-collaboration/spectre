// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <limits>
#include <memory>
#include <sstream>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTriggers/Times.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Metavariables {
  using component_list = tmpl::list<>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<DenseTrigger, tmpl::list<DenseTriggers::Times>>,
                  tmpl::pair<TimeSequence<double>,
                             TimeSequences::all_time_sequences<double>>>;
  };
};

void check_one_direction(const std::vector<double>& trigger_times,
                         const double current_time,
                         const bool expected_is_triggered,
                         const double expected_next_check,
                         const bool time_runs_forward) noexcept {
  CAPTURE(time_runs_forward);

  std::stringstream creation_string;
  creation_string.precision(std::numeric_limits<double>::max_digits10);
  creation_string << "Times:\n"
                  << "  Specified:\n"
                  << "    Values: [";
  for (auto time : trigger_times) {
    creation_string << time << ",";
  }
  creation_string << "]";
  CAPTURE(creation_string.str());

  const auto trigger = serialize_and_deserialize(
      TestHelpers::test_creation<std::unique_ptr<DenseTrigger>, Metavariables>(
          creation_string.str()));

  const Slab slab(1.0e10, 2.0e10);
  auto box = db::create<
      db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<Metavariables>,
                        Tags::TimeStepId, Tags::Time>>(
      Metavariables{}, TimeStepId(time_runs_forward, 100, slab.start()),
      current_time);
  Parallel::GlobalCache<Metavariables> cache{};
  const int array_index = 0;
  const void* component = nullptr;

  CHECK(trigger->is_ready(box, cache, array_index, component));
  CHECK_FALSE(trigger->previous_trigger_time().has_value());
  const auto is_triggered = trigger->is_triggered(box);
  CHECK(is_triggered.is_triggered == expected_is_triggered);
  CHECK(is_triggered.next_check == expected_next_check);
  if (expected_is_triggered) {
    CHECK_FALSE(trigger->previous_trigger_time().has_value());
    db::mutate<::Tags::Time>(
        make_not_null(&box),
        [](const gsl::not_null<double*> time) noexcept { *time += 0.01; });
    CHECK_FALSE(trigger->is_triggered(box).is_triggered);
    REQUIRE(trigger->previous_trigger_time().has_value());
    CHECK(trigger->previous_trigger_time().value() == current_time);
  } else {
    CHECK_FALSE(trigger->previous_trigger_time().has_value());
    db::mutate<::Tags::Time>(
        make_not_null(&box),
        [](const gsl::not_null<double*> time) noexcept { *time += 0.01; });
    CHECK_FALSE(trigger->is_triggered(box).is_triggered);
    CHECK_FALSE(trigger->previous_trigger_time().has_value());
  }
}

void check_both_directions(std::vector<double> trigger_times,
                           const double current_time,
                           const bool expected_is_triggered,
                           const double expected_next_check) noexcept {
  check_one_direction(trigger_times, current_time, expected_is_triggered,
                      expected_next_check, true);
  alg::for_each(trigger_times, [](double& t) noexcept { t = -t; });
  check_one_direction(trigger_times, -current_time, expected_is_triggered,
                      -expected_next_check, false);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.EventsAndDenseTriggers.DenseTriggers.Times",
                  "[Unit][Evolution]") {
  Parallel::register_factory_classes_with_charm<Metavariables>();

  const auto infinity = std::numeric_limits<double>::infinity();

  check_both_directions({}, 1.0, false, infinity);

  check_both_directions({1.7}, 1.6, false, 1.7);
  check_both_directions({1.7}, 1.7, true, infinity);
  check_both_directions({1.7}, 1.8, false, infinity);

  check_both_directions({1.7, 1.9}, 1.6, false, 1.7);
  check_both_directions({1.7, 1.9}, 1.7, true, 1.9);
  check_both_directions({1.7, 1.9}, 1.8, false, 1.9);
  check_both_directions({1.7, 1.9}, 1.9, true, infinity);
  check_both_directions({1.7, 1.9}, 2.0, false, infinity);
}
