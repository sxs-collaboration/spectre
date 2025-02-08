// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <optional>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/ExitCode.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/CheckpointAndExitAfterWallclock.hpp"
#include "Parallel/PhaseControl/PhaseControlTags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

struct Metavariables {
  using component_list = tmpl::list<>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<PhaseChange,
                   tmpl::list<PhaseControl::CheckpointAndExitAfterWallclock>>,
        tmpl::pair<Trigger, tmpl::list<Triggers::Always>>>;
  };

};

SPECTRE_TEST_CASE("Unit.Parallel.PhaseControl.CheckpointAndExitAfterWallclock",
                  "[Unit][Parallel]") {
  // note that the `contribute_phase_data_impl` function is currently untested
  // in this unit test, because we do not have good support for reductions in
  // the action testing framework.

  const auto created_phase_changes = TestHelpers::test_option_tag<
      PhaseControl::OptionTags::PhaseChangeAndTriggers, Metavariables>(
      " - Trigger: Always\n"
      "   PhaseChanges:\n"
      "     - CheckpointAndExitAfterWallclock:\n"
      "         WallclockHours: 0.0");

  Parallel::GlobalCache<Metavariables> cache{};

  using PhaseChangeDecisionData = tuples::tagged_tuple_from_typelist<
      PhaseControl::get_phase_change_tags<Metavariables>>;

  const PhaseControl::CheckpointAndExitAfterWallclock phase_change0(0.0);
  const PhaseControl::CheckpointAndExitAfterWallclock phase_change1(1.0);
  {
    INFO("Test initialize phase change decision data");
    PhaseChangeDecisionData phase_change_decision_data{
        Parallel::Phase::Execute, true, true, Parallel::ExitCode::Complete};
    phase_change0.initialize_phase_data<Metavariables>(
        make_not_null(&phase_change_decision_data));
    CHECK(phase_change_decision_data ==
          PhaseChangeDecisionData{std::nullopt, false, true,
                                  Parallel::ExitCode::Complete});
  }
  {
    INFO("Wallclock time < big trigger time");
    // Check behavior when a checkpoint-and-exit has been requested
    // First check case where wallclock time < trigger wallclock time, using
    // the PhaseChange with a big trigger time.
    // (this assumes the test doesn't take 1h to get here)
    PhaseChangeDecisionData phase_change_decision_data{
        std::nullopt, true, true, Parallel::ExitCode::Complete};
    const auto decision_result = phase_change1.arbitrate_phase_change(
        make_not_null(&phase_change_decision_data), Parallel::Phase::Execute,
        cache);
    CHECK(decision_result == std::nullopt);
    CHECK(phase_change_decision_data ==
          PhaseChangeDecisionData{std::nullopt, false, true,
                                  Parallel::ExitCode::Complete});
  }
  {
    INFO("Wallclock time > small trigger time");
    // Now check case where wallclock time > trigger wallclock time, using
    // the PhaseChange with a tiny trigger time.
    // (this assumes the test takes at least a few cycles to get here)
    PhaseChangeDecisionData phase_change_decision_data{
        std::nullopt, true, true, Parallel::ExitCode::Complete};
    const auto decision_result = phase_change0.arbitrate_phase_change(
        make_not_null(&phase_change_decision_data), Parallel::Phase::Execute,
        cache);
    CHECK(
        decision_result ==
        std::make_pair(Parallel::Phase::WriteCheckpoint,
                       PhaseControl::ArbitrationStrategy::RunPhaseImmediately));
    CHECK(phase_change_decision_data ==
          PhaseChangeDecisionData{Parallel::Phase::Execute, false, true,
                                  Parallel::ExitCode::Complete});
  }
  {
    INFO("Restarting from checkpoint");
    // Check behavior following the checkpoint phase
    const PhaseControl::CheckpointAndExitAfterWallclock phase_change_restart =
        serialize_and_deserialize(phase_change0);
    PhaseChangeDecisionData phase_change_decision_data{
        Parallel::Phase::Execute, false, true, Parallel::ExitCode::Complete};
    auto decision_result = phase_change_restart.arbitrate_phase_change(
        make_not_null(&phase_change_decision_data),
        Parallel::Phase::WriteCheckpoint, cache);
    CHECK(decision_result ==
          std::make_pair(
              Parallel::Phase::UpdateOptionsAtRestartFromCheckpoint,
              PhaseControl::ArbitrationStrategy::PermitAdditionalJumps));
    CHECK(phase_change_decision_data ==
          PhaseChangeDecisionData{Parallel::Phase::Execute, false, true,
                                  Parallel::ExitCode::Complete});

    // Now, from update phase, go back to Execute
    decision_result = phase_change_restart.arbitrate_phase_change(
        make_not_null(&phase_change_decision_data),
        Parallel::Phase::UpdateOptionsAtRestartFromCheckpoint, cache);
    CHECK(decision_result ==
          std::make_pair(
              Parallel::Phase::Execute,
              PhaseControl::ArbitrationStrategy::PermitAdditionalJumps));
    CHECK(phase_change_decision_data ==
          PhaseChangeDecisionData{std::nullopt, false, true,
                                  Parallel::ExitCode::Complete});
  }
  {
    INFO("Exiting after checkpoint");
    PhaseChangeDecisionData phase_change_decision_data{
        Parallel::Phase::Execute, false, true, Parallel::ExitCode::Complete};
    const auto decision_result = phase_change0.arbitrate_phase_change(
        make_not_null(&phase_change_decision_data),
        Parallel::Phase::WriteCheckpoint, cache);
    CHECK(
        decision_result ==
        std::make_pair(Parallel::Phase::Exit,
                       PhaseControl::ArbitrationStrategy::RunPhaseImmediately));
    CHECK(phase_change_decision_data ==
          PhaseChangeDecisionData{Parallel::Phase::Execute, false, true,
                                  Parallel::ExitCode::ContinueFromCheckpoint});
  }
}
