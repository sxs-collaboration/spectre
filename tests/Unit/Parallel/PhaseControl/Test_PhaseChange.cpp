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
#include "Options/Options.hpp"
#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/PhaseControl/PhaseChange.hpp"
#include "Parallel/PhaseControl/PhaseControlTags.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
template <typename PhaseChangeRegistrars>
struct TestPhaseChange;

namespace Registrars {
using TestPhaseChange = Registration::Registrar<TestPhaseChange>;
}  // namespace Registrars

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
  using initialization_tags = tmpl::list<>;
};

struct TestComponentBeta {
  using phase_dependent_action_list = tmpl::list<>;
  using chare_type = Parallel::Algorithms::Array;
  using array_index = int;
  using metavariables = Metavariables;
  using initialization_tags = tmpl::list<>;
};

namespace TestGlobalStateRecord {
bool contributed = false;
}  // namespace TestGlobalStateRecord

template <typename PhaseChangeRegistrars =
              tmpl::list<Registrars::TestPhaseChange>>
struct TestPhaseChange : public PhaseChange<PhaseChangeRegistrars> {
  TestPhaseChange() = default;
  explicit TestPhaseChange(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(TestPhaseChange);  // NOLINT

  struct Factor {
    using type = int;
    static constexpr Options::String help {"Multiplier to apply"};
  };

  using options = tmpl::list<Factor>;
  static constexpr Options::String help{"Phase change tester"};

  using argument_tags = tmpl::list<Tags::InputValue>;
  using return_tags = tmpl::list<Tags::MutatedValue>;

  using phase_change_tags_and_combines_list = tmpl::list<Tags::Request>;
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
  typename std::optional<std::pair<typename Metavariables::Phase,
                                   PhaseControl::ArbitrationStrategy>>
  arbitrate_phase_change_impl(
      const gsl::not_null<tuples::TaggedTuple<DecisionTags...>*>
          phase_change_decision_data,
      const typename Metavariables::Phase /*current_phase*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/) const {
    if (tuples::get<Tags::Request>(*phase_change_decision_data)) {
      tuples::get<Tags::Request>(*phase_change_decision_data) = false;
      return std::make_pair(
          Metavariables::Phase::PhaseA,
          PhaseControl::ArbitrationStrategy::RunPhaseImmediately);
    } else {
      return std::nullopt;
    }
  }

  void pup(PUP::er& p) override { p | stored_multiplier_; }  // NOLINT

 private:
  int stored_multiplier_ = 0;
};

template <typename PhaseChangeRegistrars>
PUP::able::PUP_ID TestPhaseChange<PhaseChangeRegistrars>::my_PUP_ID = 0;

struct Metavariables {
  using component_list = tmpl::list<TestComponentAlpha, TestComponentBeta>;

  enum class Phase { PhaseA, PhaseB };
};

SPECTRE_TEST_CASE("Unit.Parallel.PhaseControl.PhaseChange",
                  "[Unit][Parallel]") {
  // create the necessary container types
  Parallel::GlobalCache<Metavariables> cache{};
  auto box =
      db::create<db::AddSimpleTags<Tags::MutatedValue, Tags::InputValue>>(
          std::numeric_limits<double>::signaling_NaN(), -5);
  tuples::TaggedTuple<Tags::Request> phase_change_decision_data{true};

  auto phase_change = TestHelpers::test_creation<
      std::unique_ptr<PhaseChange<tmpl::list<Registrars::TestPhaseChange>>>>(
      "TestPhaseChange:\n"
      "  Factor: 3");

  CHECK(tuples::get<Tags::Request>(phase_change_decision_data));
  phase_change->initialize_phase_data(
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
      make_not_null(&phase_change_decision_data), Metavariables::Phase::PhaseB,
      cache);
  CHECK_FALSE(first_decision.has_value());

  tuples::get<Tags::Request>(phase_change_decision_data) = true;
  const auto second_decision = phase_change->arbitrate_phase_change(
      make_not_null(&phase_change_decision_data), Metavariables::Phase::PhaseB,
      cache);
  CHECK(second_decision.value() ==
        std::make_pair(Metavariables::Phase::PhaseA,
                       PhaseControl::ArbitrationStrategy::RunPhaseImmediately));
  CHECK_FALSE(tuples::get<Tags::Request>(phase_change_decision_data));
}
}  // namespace
