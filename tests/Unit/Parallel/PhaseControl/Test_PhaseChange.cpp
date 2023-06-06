// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Framework/TestCreation.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/PhaseChange.hpp"
#include "Parallel/PhaseControl/PhaseControlTags.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
namespace Tags {
struct Request {
  using type = bool;

  using combine_method = funcl::Or<>;
  using main_combine_method = funcl::Or<>;
};

struct MutatedValue : db::SimpleTag {
  using type = double;
};

struct InputValue: db::SimpleTag {
  using type = int;
};
}  // namespace Tags

struct Metavariables;

struct TestComponentAlpha {
  using phase_dependent_action_list = tmpl::list<>;
  using chare_type = Parallel::Algorithms::Array;
  using array_index = int;
  using metavariables = Metavariables;
  using simple_tags_from_options = tmpl::list<>;
};

struct TestComponentBeta {
  using phase_dependent_action_list = tmpl::list<>;
  using chare_type = Parallel::Algorithms::Array;
  using array_index = int;
  using metavariables = Metavariables;
  using simple_tags_from_options = tmpl::list<>;
};

namespace TestGlobalStateRecord {
bool contributed = false;
}  // namespace TestGlobalStateRecord

struct TestPhaseChange : public PhaseChange {
  TestPhaseChange() = default;
  explicit TestPhaseChange(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  WRAPPED_PUPable_decl_template(TestPhaseChange);  // NOLINT
#pragma GCC diagnostic pop

  struct Factor {
    using type = int;
    static constexpr Options::String help {"Multiplier to apply"};
  };

  using options = tmpl::list<Factor>;
  static constexpr Options::String help{"Phase change tester"};

  using argument_tags = tmpl::list<Tags::InputValue>;
  using return_tags = tmpl::list<Tags::MutatedValue>;

  template <typename Metavariables>
  using participating_components = tmpl::list<TestComponentAlpha>;

  explicit TestPhaseChange(int multiplier) : stored_multiplier_{multiplier} {}

  template <typename... DecisionTags>
  void initialize_phase_data_impl(
      const gsl::not_null<tuples::TaggedTuple<DecisionTags...>*>
          phase_change_decision_data) const {
    tuples::get<Tags::Request>(*phase_change_decision_data) = false;
  }

  template <typename ParallelComponent, typename Metavariables,
            typename ArrayIndex>
  void contribute_phase_data_impl(
      const gsl::not_null<double*> mutated_value, const int input_value,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/) const {
    TestGlobalStateRecord::contributed = true;
    *mutated_value =
        static_cast<double>(square(input_value) * stored_multiplier_);
  }

  template <typename... DecisionTags, typename Metavariables>
  typename std::optional<
      std::pair<Parallel::Phase, PhaseControl::ArbitrationStrategy>>
  arbitrate_phase_change_impl(
      const gsl::not_null<tuples::TaggedTuple<DecisionTags...>*>
          phase_change_decision_data,
      const Parallel::Phase /*current_phase*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/) const {
    if (tuples::get<Tags::Request>(*phase_change_decision_data)) {
      tuples::get<Tags::Request>(*phase_change_decision_data) = false;
      return std::make_pair(
          Parallel::Phase::Solve,
          PhaseControl::ArbitrationStrategy::RunPhaseImmediately);
    } else {
      return std::nullopt;
    }
  }

  void pup(PUP::er& p) override { p | stored_multiplier_; }  // NOLINT

 private:
  int stored_multiplier_ = 0;
};

PUP::able::PUP_ID TestPhaseChange::my_PUP_ID = 0;

struct Metavariables {
  using component_list = tmpl::list<TestComponentAlpha, TestComponentBeta>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<PhaseChange, tmpl::list<TestPhaseChange>>>;
  };

};

SPECTRE_TEST_CASE("Unit.Parallel.PhaseControl.PhaseChange",
                  "[Unit][Parallel]") {
  // create the necessary container types
  Parallel::GlobalCache<Metavariables> cache{};
  auto box =
      db::create<db::AddSimpleTags<Tags::MutatedValue, Tags::InputValue>>(
          std::numeric_limits<double>::signaling_NaN(), -5);
  tuples::TaggedTuple<Tags::Request> phase_change_decision_data{true};

  auto phase_change =
      TestHelpers::test_creation<std::unique_ptr<PhaseChange>, Metavariables>(
          "TestPhaseChange:\n"
          "  Factor: 3");

  CHECK(tuples::get<Tags::Request>(phase_change_decision_data));
  phase_change->initialize_phase_data<Metavariables>(
      make_not_null(&phase_change_decision_data));
  CHECK_FALSE(tuples::get<Tags::Request>(phase_change_decision_data));

  CHECK_FALSE(TestGlobalStateRecord::contributed);
  phase_change->contribute_phase_data<TestComponentBeta>(make_not_null(&box),
                                                         cache, 0);
  CHECK_FALSE(TestGlobalStateRecord::contributed);
  phase_change->contribute_phase_data<TestComponentAlpha>(make_not_null(&box),
                                                          cache, 0);
  CHECK(TestGlobalStateRecord::contributed);
  CHECK(db::get<Tags::MutatedValue>(box) == 75);

  const auto first_decision = phase_change->arbitrate_phase_change(
      make_not_null(&phase_change_decision_data), Parallel::Phase::Evolve,
      cache);
  CHECK_FALSE(first_decision.has_value());

  tuples::get<Tags::Request>(phase_change_decision_data) = true;
  const auto second_decision = phase_change->arbitrate_phase_change(
      make_not_null(&phase_change_decision_data), Parallel::Phase::Evolve,
      cache);
  CHECK(second_decision.value() ==
        std::make_pair(Parallel::Phase::Solve,
                       PhaseControl::ArbitrationStrategy::RunPhaseImmediately));
  CHECK_FALSE(tuples::get<Tags::Request>(phase_change_decision_data));
}
}  // namespace
