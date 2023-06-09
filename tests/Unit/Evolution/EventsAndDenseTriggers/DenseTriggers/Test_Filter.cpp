// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <memory>
#include <sstream>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTriggers/Filter.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTriggers/Times.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/EventsAndDenseTriggers/DenseTriggers/TestTrigger.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/Triggers/TimeCompares.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
class TestTrigger : public Trigger {
 public:
  TestTrigger() = default;
  explicit TestTrigger(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  WRAPPED_PUPable_decl_template(TestTrigger);  // NOLINT
#pragma GCC diagnostic pop

  struct Result {
    using type = bool;
    constexpr static Options::String help = "Result";
  };

  using options = tmpl::list<Result>;
  constexpr static Options::String help = "help";

  explicit TestTrigger(const bool result) : result_(result) {}

  using argument_tags = tmpl::list<>;
  bool operator()() const { return result_; }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    Trigger::pup(p);
    p | result_;
  }

 private:
  bool result_{};
};

PUP::able::PUP_ID TestTrigger::my_PUP_ID = 0;  // NOLINT

struct Metavariables {
  using component_list = tmpl::list<>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<DenseTrigger,
                   tmpl::list<DenseTriggers::Filter,
                              TestHelpers::DenseTriggers::TestTrigger>>,
        tmpl::pair<TimeSequence<double>,
                   TimeSequences::all_time_sequences<double>>,
        tmpl::pair<Trigger, tmpl::list<TestTrigger>>>;
  };
};

void check(const std::optional<bool>& expected_is_triggered,
           const std::optional<double>& expected_next_check,
           const std::optional<bool>& dense_is_triggered,
           const bool non_dense_is_triggered) {
  using NotReady = TestHelpers::DenseTriggers::TestTrigger::NotReady;
  std::stringstream creation_string;
  creation_string << std::boolalpha;
  creation_string << "Filter:\n"
                  << "  Trigger:\n"
                  << "    TestTrigger:\n"
                  << "      IsTriggered: "
                  << NotReady::format(dense_is_triggered) << "\n"
                  << "      NextCheck: "
                  << NotReady::format(expected_next_check) << "\n"
                  << "  Filter:\n"
                  << "    TestTrigger:\n"
                  << "      Result: " << non_dense_is_triggered;
  CAPTURE(creation_string.str());
  const auto box = db::create<db::AddSimpleTags<
      Parallel::Tags::MetavariablesImpl<Metavariables>, ::Tags::Time>>(
      Metavariables{}, 0.0);
  Parallel::GlobalCache<Metavariables> cache{};
  const int array_index = 0;
  const void* component = nullptr;
  const auto trigger = serialize_and_deserialize(
      TestHelpers::test_creation<std::unique_ptr<DenseTrigger>, Metavariables>(
          creation_string.str()));
  CHECK(trigger->is_triggered(box, cache, array_index, component) ==
        expected_is_triggered);
  CHECK(trigger->next_check_time(box, cache, array_index, component) ==
        expected_next_check);
}

struct ExampleMetavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<DenseTrigger, tmpl::list<DenseTriggers::Filter,
                                                      DenseTriggers::Times>>,
                  tmpl::pair<TimeSequence<double>,
                             TimeSequences::all_time_sequences<double>>,
                  tmpl::pair<Trigger, tmpl::list<Triggers::TimeCompares>>>;
  };
};

void example() {
  const std::string input = R"(
# [example]
Filter:
  Trigger:
    Times:
      EvenlySpaced:
        Offset: 0
        Interval: 10
  Filter:
    TimeCompares:
      Comparison: GreaterThanOrEqualTo
      Value: 100
# [example]
)";
  const auto trigger = TestHelpers::test_creation<std::unique_ptr<DenseTrigger>,
                                                  ExampleMetavariables>(input);
  const auto run_trigger = [&trigger](const double time) {
    const auto box = db::create<db::AddSimpleTags<
        Parallel::Tags::MetavariablesImpl<ExampleMetavariables>, ::Tags::Time,
        ::Tags::TimeStepId>>(ExampleMetavariables{}, time,
                             TimeStepId(true, 0, Slab(0., 1.).start()));
    Parallel::GlobalCache<Metavariables> cache{};
    const int array_index = 0;
    const void* component = nullptr;
    return *trigger->is_triggered(box, cache, array_index, component);
  };
  CHECK(not run_trigger(90.));
  CHECK(not run_trigger(95.));
  CHECK(run_trigger(100.));
  CHECK(not run_trigger(105.));
  CHECK(run_trigger(110.));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.EventsAndDenseTriggers.DenseTriggers.Filter",
                  "[Unit][Evolution]") {
  register_factory_classes_with_charm<Metavariables>();

  for (const auto& next_check : {std::optional{3.5}, std::optional<double>{}}) {
    check(std::nullopt, next_check, std::nullopt, false);
    check(std::nullopt, next_check, std::nullopt, true);
    check(false, next_check, false, false);
    check(false, next_check, false, true);
    check(false, next_check, true, false);
    check(true, next_check, true, true);
  }

  example();
}
