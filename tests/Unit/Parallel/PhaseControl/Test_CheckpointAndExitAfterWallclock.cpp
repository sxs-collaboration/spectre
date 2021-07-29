// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <optional>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Framework/TestCreation.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
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
    using factory_classes =
        tmpl::map<tmpl::pair<Trigger, tmpl::list<Triggers::Always>>>;
  };

  enum class Phase {
    PhaseA,
    WriteCheckpoint,
    UpdateOptionsAtRestartFromCheckpoint,
    Exit
  };
};

SPECTRE_TEST_CASE("Unit.Parallel.PhaseControl.CheckpointAndExitAfterWallclock",
                  "[Unit][Parallel]") {
  // note that the `contribute_phase_data_impl` function is currently untested
  // in this unit test, because we do not have good support for reductions in
  // the action testing framework.
  using phase_changes = tmpl::list<
      PhaseControl::Registrars::CheckpointAndExitAfterWallclock<Metavariables>>;

  const auto created_phase_changes = TestHelpers::test_option_tag<
      PhaseControl::OptionTags::PhaseChangeAndTriggers<phase_changes>,
      Metavariables>(
      " - - Always:\n"
      "   - - CheckpointAndExitAfterWallclock:\n"
      "         WallclockHours: 0.0");

  Parallel::GlobalCache<Metavariables> cache{};

  using phase_change_decision_data_type = tuples::tagged_tuple_from_typelist<
      PhaseControl::get_phase_change_tags<phase_changes>>;
  phase_change_decision_data_type phase_change_decision_data{
      Metavariables::Phase::PhaseA, true, 1.0, true};

  const PhaseControl::CheckpointAndExitAfterWallclock<Metavariables>
      phase_change0(0.0);
  const PhaseControl::CheckpointAndExitAfterWallclock<Metavariables>
      phase_change1(1.0);
  {
    INFO("Test initialize phase change decision data");
    phase_change0.initialize_phase_data(
        make_not_null(&phase_change_decision_data));
    // extra parens in the check prevent Catch from trying to stream the tuple
    CHECK((phase_change_decision_data ==
           phase_change_decision_data_type{std::nullopt, std::nullopt, false,
                                           true}));
  }
  {
    INFO("Test arbitrate phase control");
    // Check behavior when a checkpoint-and-exit has been requested
    // First check case where wallclock time < trigger wallclock time, using
    // the PhaseChange with a big trigger time.
    // (this assumes the test doesn't take 1h to get here)
    phase_change_decision_data =
        phase_change_decision_data_type{std::nullopt, std::nullopt, true, true};
    auto decision_result = phase_change1.arbitrate_phase_change(
        make_not_null(&phase_change_decision_data),
        Metavariables::Phase::PhaseA, cache);
    CHECK((decision_result == std::nullopt));
    CHECK((phase_change_decision_data ==
           phase_change_decision_data_type{std::nullopt, std::nullopt, false,
                                           true}));

    // Now check case where wallclock time < trigger wallclock time, using
    // the PhaseChange with a tiny trigger time.
    // (this assumes the test takes at least a few cycles to get here)
    phase_change_decision_data =
        phase_change_decision_data_type{std::nullopt, std::nullopt, true, true};
    decision_result = phase_change0.arbitrate_phase_change(
        make_not_null(&phase_change_decision_data),
        Metavariables::Phase::PhaseA, cache);
    CHECK((decision_result ==
           std::make_pair(
               Metavariables::Phase::WriteCheckpoint,
               PhaseControl::ArbitrationStrategy::RunPhaseImmediately)));
    // It's impossible to know what the elapsed wallclock time will be, so we
    // check the tags one by one...
    CHECK((tuples::get<
               PhaseControl::Tags::RestartPhase<typename Metavariables::Phase>>(
               phase_change_decision_data) == Metavariables::Phase::PhaseA));
    // Check recorded time in range: 0 second < time < 1 second
    // (this assumes test run duration falls in this time window)
    CHECK(tuples::get<PhaseControl::Tags::WallclockHoursAtCheckpoint>(
              phase_change_decision_data) > 0.0);
    const double one_second = 1.0 / 3600.0;
    CHECK(tuples::get<PhaseControl::Tags::WallclockHoursAtCheckpoint>(
              phase_change_decision_data) < one_second);
    CHECK(tuples::get<PhaseControl::Tags::CheckpointAndExitRequested>(
              phase_change_decision_data) == false);

    // Check behavior following the checkpoint phase
    // First check case where wallclock time < recorded time, which corresponds
    // to restarting from a checkpoint. Should update options next.
    // (this assumes the test doesn't take 1h to get here)
    phase_change_decision_data = phase_change_decision_data_type{
        Metavariables::Phase::PhaseA, 1.0, false, true};
    decision_result = phase_change0.arbitrate_phase_change(
        make_not_null(&phase_change_decision_data),
        Metavariables::Phase::WriteCheckpoint, cache);
    CHECK((decision_result ==
           std::make_pair(
               Metavariables::Phase::UpdateOptionsAtRestartFromCheckpoint,
               PhaseControl::ArbitrationStrategy::PermitAdditionalJumps)));
    CHECK((phase_change_decision_data ==
           phase_change_decision_data_type{Metavariables::Phase::PhaseA, 1.0,
                                           false, true}));

    // Now, from update phase, go back to PhaseA
    decision_result = phase_change0.arbitrate_phase_change(
        make_not_null(&phase_change_decision_data),
        Metavariables::Phase::UpdateOptionsAtRestartFromCheckpoint, cache);
    CHECK((decision_result ==
           std::make_pair(
               Metavariables::Phase::PhaseA,
               PhaseControl::ArbitrationStrategy::PermitAdditionalJumps)));
    CHECK((phase_change_decision_data ==
           phase_change_decision_data_type{std::nullopt, std::nullopt, false,
                                           true}));

    // Now check case where wallclock time > recorded time, which corresponds to
    // having just written a checkpoint. We want to exit now.
    // (this assumes the test takes at least a few cycles to get here)
    phase_change_decision_data = phase_change_decision_data_type{
        Metavariables::Phase::PhaseA, 1e-15, false, true};
    decision_result = phase_change0.arbitrate_phase_change(
        make_not_null(&phase_change_decision_data),
        Metavariables::Phase::WriteCheckpoint, cache);
    CHECK((decision_result ==
           std::make_pair(
               Metavariables::Phase::Exit,
               PhaseControl::ArbitrationStrategy::RunPhaseImmediately)));
    CHECK((phase_change_decision_data ==
           phase_change_decision_data_type{Metavariables::Phase::PhaseA, 1e-15,
                                           false, true}));
  }
}
