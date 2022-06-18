// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <optional>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Framework/TestCreation.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/PhaseControlTags.hpp"
#include "Parallel/PhaseControl/VisitAndReturn.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Metavariables {
  using component_list = tmpl::list<>;


  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<
            PhaseChange,
            tmpl::list<PhaseControl::VisitAndReturn<Parallel::Phase::Evolve>,
                       PhaseControl::VisitAndReturn<Parallel::Phase::Execute>>>,
        tmpl::pair<Trigger, tmpl::list<Triggers::Always>>>;
  };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Parallel.PhaseControl.VisitAndReturn",
                  "[Unit][Parallel]") {
  // note that the `contribute_phase_data_impl` function is currently untested
  // in this unit test, because we do not have good support for reductions in
  // the action testing framework. These are tested in the integration test
  // `Parallel/Test_AlgorithmPhaseControl.cpp`
  Parallel::GlobalCache<Metavariables> cache{};

  const auto created_phase_changes = TestHelpers::test_option_tag<
      PhaseControl::OptionTags::PhaseChangeAndTriggers, Metavariables>(
      " - - Always:\n"
      "   - - VisitAndReturn(Evolve):\n"
      "     - VisitAndReturn(Execute):");
  using phase_change_decision_data_type = tuples::tagged_tuple_from_typelist<
      PhaseControl::get_phase_change_tags<Metavariables>>;

  phase_change_decision_data_type phase_change_decision_data{
      Parallel::Phase::Solve, true, Parallel::Phase::Solve, true, true};
  const auto& first_phase_change = created_phase_changes[0].second[0];
  const auto& second_phase_change = created_phase_changes[0].second[1];
  {
    INFO("Test initialize phase change decision data");
    first_phase_change->initialize_phase_data<Metavariables>(
        make_not_null(&phase_change_decision_data));
    // extra parens in the check prevent Catch from trying to stream the tuple
    CHECK((phase_change_decision_data ==
           phase_change_decision_data_type{
               std::nullopt, false, Parallel::Phase::Solve, true, true}));

    second_phase_change->initialize_phase_data<Metavariables>(
        make_not_null(&phase_change_decision_data));
    CHECK((phase_change_decision_data ==
           phase_change_decision_data_type{std::nullopt, false, std::nullopt,
                                           false, true}));

    using first_phase_change_tuple_type = tuples::TaggedTuple<
        PhaseControl::Tags::ReturnPhase<Parallel::Phase::Evolve>,
        PhaseControl::Tags::TemporaryPhaseRequested<Parallel::Phase::Evolve>>;
    first_phase_change_tuple_type tuple_for_first_phase_change{
        Parallel::Phase::Solve, true};
    PhaseControl::VisitAndReturn<Parallel::Phase::Evolve>{}
        .initialize_phase_data_impl(
            make_not_null(&tuple_for_first_phase_change));
    CHECK((tuple_for_first_phase_change ==
           first_phase_change_tuple_type{std::nullopt, false}));
  }
  {
    INFO("Test arbitrate phase control");
    // In this test, we trace through the set of states that occur when
    // resolving two simultaneous phase requests. This is a superset of states
    // that occur in a single phase request
    phase_change_decision_data = phase_change_decision_data_type{
        std::nullopt, true, std::nullopt, true, true};
    auto decision_result = first_phase_change->arbitrate_phase_change(
        make_not_null(&phase_change_decision_data), Parallel::Phase::Execute,
        cache);
    // extra parens in the check prevent Catch from trying to stream the tuple
    CHECK((decision_result ==
           std::make_pair(
               Parallel::Phase::Evolve,
               PhaseControl::ArbitrationStrategy::RunPhaseImmediately)));
    CHECK((phase_change_decision_data ==
           phase_change_decision_data_type{Parallel::Phase::Execute, false,
                                           std::nullopt, true, true}));

    // check a different starting phase
    phase_change_decision_data = phase_change_decision_data_type{
        std::nullopt, true, std::nullopt, true, true};
    decision_result = first_phase_change->arbitrate_phase_change(
        make_not_null(&phase_change_decision_data), Parallel::Phase::Solve,
        cache);
    CHECK((decision_result ==
           std::make_pair(
               Parallel::Phase::Evolve,
               PhaseControl::ArbitrationStrategy::RunPhaseImmediately)));
    CHECK((phase_change_decision_data ==
           phase_change_decision_data_type{Parallel::Phase::Solve, false,
                                           std::nullopt, true, true}));

    decision_result = first_phase_change->arbitrate_phase_change(
        make_not_null(&phase_change_decision_data), Parallel::Phase::Evolve,
        cache);
    CHECK((decision_result ==
           std::make_pair(
               Parallel::Phase::Solve,
               PhaseControl::ArbitrationStrategy::PermitAdditionalJumps)));
    CHECK((phase_change_decision_data ==
           phase_change_decision_data_type{std::nullopt, false, std::nullopt,
                                           true, true}));

    decision_result = second_phase_change->arbitrate_phase_change(
        make_not_null(&phase_change_decision_data), Parallel::Phase::Solve,
        cache);
    CHECK((decision_result ==
           std::make_pair(
               Parallel::Phase::Execute,
               PhaseControl::ArbitrationStrategy::RunPhaseImmediately)));
    CHECK((phase_change_decision_data ==
           phase_change_decision_data_type{
               std::nullopt, false, Parallel::Phase::Solve, false, true}));

    decision_result = PhaseControl::VisitAndReturn<Parallel::Phase::Execute>{}
                          .arbitrate_phase_change_impl(
                              make_not_null(&phase_change_decision_data),
                              Parallel::Phase::Execute, cache);
    CHECK((decision_result ==
           std::make_pair(
               Parallel::Phase::Solve,
               PhaseControl::ArbitrationStrategy::PermitAdditionalJumps)));
    CHECK((phase_change_decision_data ==
           phase_change_decision_data_type{std::nullopt, false, std::nullopt,
                                           false, true}));

    decision_result = second_phase_change->arbitrate_phase_change(
        make_not_null(&phase_change_decision_data), Parallel::Phase::Solve,
        cache);
    CHECK((decision_result == std::nullopt));
    CHECK((phase_change_decision_data ==
           phase_change_decision_data_type{std::nullopt, false, std::nullopt,
                                           false, true}));
  }
}
