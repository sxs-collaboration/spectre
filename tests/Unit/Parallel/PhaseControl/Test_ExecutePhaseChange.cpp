// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>

#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/PhaseControl/ExecutePhaseChange.hpp"
#include "Parallel/PhaseControl/InitializePhaseChangeDecisionData.hpp"
#include "Parallel/PhaseControl/PhaseChange.hpp"
#include "Parallel/PhaseControl/PhaseControlTags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

namespace TestTags {
template <PhaseControl::ArbitrationStrategy, size_t Index>
struct Request {
  using type = bool;

  using combine_method = funcl::Or<>;
  using main_combine_method = funcl::Or<>;
};
}  // namespace TestTags

template <PhaseControl::ArbitrationStrategy Strategy, size_t Index>
struct TestPhaseChange : public PhaseChange {
  TestPhaseChange() = default;
  explicit TestPhaseChange(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(TestPhaseChange);  // NOLINT
  static std::string name() {
    if constexpr (Strategy ==
                  PhaseControl::ArbitrationStrategy::RunPhaseImmediately) {
      if constexpr (Index == 0_st) {
        return "TestPhaseChange(RunPhaseImmediately, 0)";
      } else {
        return "TestPhaseChange(RunPhaseImmediately, 1)";
      }
    } else {
      if constexpr (Index == 0_st) {
        return "TestPhaseChange(PermitAdditionalJumps, 0)";
      } else if constexpr (Index == 1_st) {
        return "TestPhaseChange(PermitAdditionalJumps, 1)";
      } else {
        return "TestPhaseChange(PermitAdditionalJumps, 2)";
      }
    }
  }

  using options = tmpl::list<>;
  static constexpr Options::String help{"Phase change tester"};

  using argument_tags = tmpl::list<>;
  using return_tags = tmpl::list<>;

  using phase_change_tags_and_combines =
      tmpl::list<TestTags::Request<Strategy, Index>>;
  template <typename Metavariables>
  using participating_components = tmpl::list<>;

  template <typename... DecisionTags>
  void initialize_phase_data_impl(
      const gsl::not_null<tuples::TaggedTuple<DecisionTags...>*>
          phase_change_decision_data) const {
    tuples::get<TestTags::Request<Strategy, Index>>(
        *phase_change_decision_data) =
        (Index % 2 == 0) xor
        (Strategy == PhaseControl::ArbitrationStrategy::RunPhaseImmediately);
  }

  template <typename ParallelComponent, typename Metavariables,
            typename ArrayIndex>
  void contribute_phase_data_impl(
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/) const {}

  template <typename... DecisionTags, typename Metavariables>
  typename std::optional<
      std::pair<Parallel::Phase, PhaseControl::ArbitrationStrategy>>
  arbitrate_phase_change_impl(
      const gsl::not_null<tuples::TaggedTuple<DecisionTags...>*>
          phase_change_decision_data,
      const Parallel::Phase /*current_phase*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/) const {
    if (tuples::get<TestTags::Request<Strategy, Index>>(
            *phase_change_decision_data)) {
      tuples::get<TestTags::Request<Strategy, Index>>(
          *phase_change_decision_data) = false;
      // Choose a unique phase, all after the first phase, for each choice of
      // the template parameters.
      return std::make_pair(
          gsl::at(Metavariables::default_phase_order,
                  1_st + static_cast<size_t>(Strategy) + Index * 2_st),
          Strategy);
    } else {
      return std::nullopt;
    }
  }

  void pup(PUP::er& /*p*/) override {}  // NOLINT
};

template <PhaseControl::ArbitrationStrategy Strategy, size_t Index>
PUP::able::PUP_ID TestPhaseChange<Strategy, Index>::my_PUP_ID = 0;

struct Metavariables {
  using component_list = tmpl::list<>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<
        PhaseChange,
        tmpl::list<
            TestPhaseChange<
                PhaseControl::ArbitrationStrategy::RunPhaseImmediately, 0_st>,
            TestPhaseChange<
                PhaseControl::ArbitrationStrategy::RunPhaseImmediately, 1_st>,
            TestPhaseChange<
                PhaseControl::ArbitrationStrategy::PermitAdditionalJumps, 0_st>,
            TestPhaseChange<
                PhaseControl::ArbitrationStrategy::PermitAdditionalJumps, 1_st>,
            TestPhaseChange<
                PhaseControl::ArbitrationStrategy::PermitAdditionalJumps,
                2_st>>>>;
  };
  using const_global_cache_tags =
      tmpl::list<PhaseControl::Tags::PhaseChangeAndTriggers>;

  static constexpr std::array<Parallel::Phase, 7> default_phase_order{
      {Parallel::Phase::Register, Parallel::Phase::Testing,
       Parallel::Phase::Solve, Parallel::Phase::ImportInitialData,
       Parallel::Phase::Evolve, Parallel::Phase::Execute,
       Parallel::Phase::Cleanup}};
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Parallel.PhaseControl.ExecutePhaseChange",
                  "[Unit][Parallel]") {
  // for a test of the `ExecutePhaseChange` action, see
  // `tests/Unit/Parallel/Test_AlgorithmPhaseControl.hpp`. Currently, there is
  // no way to test interactions with the `Main` chare via the action testing
  // framework.
  using phase_change_decision_data_type = tuples::TaggedTuple<
      ::TestTags::Request<
          PhaseControl::ArbitrationStrategy::RunPhaseImmediately, 0_st>,
      ::TestTags::Request<
          PhaseControl::ArbitrationStrategy::PermitAdditionalJumps, 0_st>,
      ::TestTags::Request<
          PhaseControl::ArbitrationStrategy::RunPhaseImmediately, 1_st>,
      ::TestTags::Request<
          PhaseControl::ArbitrationStrategy::PermitAdditionalJumps, 1_st>,
      ::TestTags::Request<
          PhaseControl::ArbitrationStrategy::PermitAdditionalJumps, 2_st>,
      PhaseControl::TagsAndCombines::UsePhaseChangeArbitration>;

  phase_change_decision_data_type phase_change_data{true, true, true,
                                                    true, true, true};

  std::vector<std::unique_ptr<PhaseChange>> phase_change_vector;
  phase_change_vector.reserve(5);
  phase_change_vector.emplace_back(
      std::make_unique<TestPhaseChange<
          PhaseControl::ArbitrationStrategy::RunPhaseImmediately, 0_st>>());
  phase_change_vector.emplace_back(
      std::make_unique<TestPhaseChange<
          PhaseControl::ArbitrationStrategy::PermitAdditionalJumps, 0_st>>());
  phase_change_vector.emplace_back(
      std::make_unique<TestPhaseChange<
          PhaseControl::ArbitrationStrategy::RunPhaseImmediately, 1_st>>());
  phase_change_vector.emplace_back(
      std::make_unique<TestPhaseChange<
          PhaseControl::ArbitrationStrategy::PermitAdditionalJumps, 1_st>>());
  phase_change_vector.emplace_back(
      std::make_unique<TestPhaseChange<
          PhaseControl::ArbitrationStrategy::PermitAdditionalJumps, 2_st>>());

  using phase_change_and_triggers = PhaseControl::Tags::PhaseChangeAndTriggers;
  phase_change_and_triggers::type vector_of_triggers_and_phase_changes;
  vector_of_triggers_and_phase_changes.push_back(
      {static_cast<std::unique_ptr<Trigger>>(
           std::make_unique<Triggers::Always>()),
       std::move(phase_change_vector)});

  Parallel::MutableGlobalCache<Metavariables> mutable_cache{};
  Parallel::GlobalCache<Metavariables> global_cache{
      tuples::TaggedTuple<phase_change_and_triggers>{
          std::move(vector_of_triggers_and_phase_changes)},
      &mutable_cache};

  {
    INFO("Initialize phase change decision data");
    PhaseControl::initialize_phase_change_decision_data(
        make_not_null(&phase_change_data), global_cache);
    // checking against the formula:
    // (Index % 2 == 0) xor
    // (Strategy == PhaseControl::ArbitrationStrategy::RunPhaseImmediately)
    CHECK(phase_change_data == phase_change_decision_data_type{
                                   false, true, true, false, true, false});
  }
  {
    INFO("Arbitrate based on phase change decision data");
    // if used, the phase change objects correspond to phases:
    // Testing, Solve, ImportInitialData, Evolve, Cleanup  (in that
    // order)
    phase_change_data = phase_change_decision_data_type{false, false, false,
                                                        false, false, false};
    CHECK(SINGLE_ARG(PhaseControl::arbitrate_phase_change(
              make_not_null(&phase_change_data), Parallel::Phase::Register,
              global_cache)) == std::nullopt);
    CHECK(phase_change_data == SINGLE_ARG(phase_change_decision_data_type{
                                   false, false, false, false, false, false}));

    phase_change_data = phase_change_decision_data_type{false, false, false,
                                                        false, false, true};
    CHECK(SINGLE_ARG(PhaseControl::arbitrate_phase_change(
              make_not_null(&phase_change_data), Parallel::Phase::Register,
              global_cache)) == Parallel::Phase::Register);
    CHECK(phase_change_data == SINGLE_ARG(phase_change_decision_data_type{
                                   false, false, false, false, false, false}));

    phase_change_data =
        phase_change_decision_data_type{true, false, true, true, false, true};
    // test the versions that use `RunPhaseImmediately`, so the phase jumps to
    // those phases without evaluating the other phase change objects
    CHECK(SINGLE_ARG(PhaseControl::arbitrate_phase_change(
              make_not_null(&phase_change_data), Parallel::Phase::Register,
              global_cache)) == Parallel::Phase::Testing);
    CHECK(phase_change_data == SINGLE_ARG(phase_change_decision_data_type{
                                   false, false, true, true, false, false}));
    phase_change_data =
        phase_change_decision_data_type{true, false, true, true, false, true};
    CHECK(SINGLE_ARG(PhaseControl::arbitrate_phase_change(
              make_not_null(&phase_change_data), Parallel::Phase::Register,
              global_cache)) == Parallel::Phase::Testing);
    CHECK(phase_change_data == SINGLE_ARG(phase_change_decision_data_type{
                                   false, false, true, true, false, false}));
    CHECK(SINGLE_ARG(PhaseControl::arbitrate_phase_change(
              make_not_null(&phase_change_data), Parallel::Phase::Register,
              global_cache)) == Parallel::Phase::ImportInitialData);
    CHECK(phase_change_data == SINGLE_ARG(phase_change_decision_data_type{
                                   false, false, false, true, false, false}));
    CHECK(SINGLE_ARG(PhaseControl::arbitrate_phase_change(
              make_not_null(&phase_change_data), Parallel::Phase::Register,
              global_cache)) == Parallel::Phase::Evolve);
    CHECK(phase_change_data == SINGLE_ARG(phase_change_decision_data_type{
                                   false, false, false, false, false, false}));

    // test the versions that use `PermitAdditionalJumps`, so each are evaluated
    // and the phase that is run is the last in the sequence, or the first
    // `RunPhaseImmediately` that is encountered.
    phase_change_data =
        phase_change_decision_data_type{false, true, false, true, false, true};
    CHECK(SINGLE_ARG(PhaseControl::arbitrate_phase_change(
              make_not_null(&phase_change_data), Parallel::Phase::Register,
              global_cache)) == Parallel::Phase::Evolve);
    CHECK(phase_change_data == SINGLE_ARG(phase_change_decision_data_type{
                                   false, false, false, false, false, false}));

    phase_change_data =
        phase_change_decision_data_type{false, true, false, true, true, true};
    CHECK(SINGLE_ARG(PhaseControl::arbitrate_phase_change(
              make_not_null(&phase_change_data), Parallel::Phase::Register,
              global_cache)) == Parallel::Phase::Cleanup);
    CHECK(phase_change_data == SINGLE_ARG(phase_change_decision_data_type{
                                   false, false, false, false, false, false}));

    // check that the result is the same starting from any phase
    for (size_t i = 0; i < 7; ++i) {
      phase_change_data = phase_change_decision_data_type{false, true,  true,
                                                          false, false, true};
      CHECK(SINGLE_ARG(PhaseControl::arbitrate_phase_change(
                make_not_null(&phase_change_data),
                gsl::at(Metavariables::default_phase_order, i),
                global_cache)) == Parallel::Phase::ImportInitialData);
      CHECK(phase_change_data ==
            SINGLE_ARG(phase_change_decision_data_type{false, false, false,
                                                       false, false, false}));
    }
  }
}
