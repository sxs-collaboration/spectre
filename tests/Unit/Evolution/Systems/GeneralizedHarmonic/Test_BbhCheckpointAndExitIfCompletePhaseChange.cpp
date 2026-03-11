// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <utility>

#include "Evolution/Systems/GeneralizedHarmonic/Bbh/PhaseControl/CheckpointAndExitIfComplete.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/PhaseChange.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Metavariables {
  using component_list = tmpl::list<>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<
        PhaseChange,
        tmpl::list<gh::bbh::phase_control::CheckpointAndExitIfComplete>>>;
  };
};
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.GeneralizedHarmonic.BbhCheckpointAndExitIfCompletePhaseChange",
    "[Unit][Evolution]") {
  // `contribute_phase_data_impl` is not tested directly because reduction
  // contributions are currently not supported in this unit-test style.
  const Parallel::GlobalCache<Metavariables> cache{};
  const gh::bbh::phase_control::CheckpointAndExitIfComplete phase_change{};
  using PhaseChangeDecisionData = tuples::TaggedTuple<
      gh::bbh::phase_control::Tags::CheckpointRequested,
      gh::bbh::phase_control::Tags::ExitAfterWriteCheckpoint>;

  {
    INFO("Initialize decision data");
    PhaseChangeDecisionData phase_change_decision_data{true, true};
    phase_change.initialize_phase_data<Metavariables>(
        make_not_null(&phase_change_decision_data));
    CHECK(phase_change_decision_data == PhaseChangeDecisionData{false, false});
  }
  {
    INFO("No request means no phase change");
    PhaseChangeDecisionData phase_change_decision_data{false, false};
    const auto decision_result = phase_change.arbitrate_phase_change(
        make_not_null(&phase_change_decision_data), Parallel::Phase::Evolve,
        cache);
    CHECK(decision_result == std::nullopt);
    CHECK(phase_change_decision_data == PhaseChangeDecisionData{false, false});
  }
  {
    INFO("Completion request in Evolve jumps to WriteCheckpoint");
    PhaseChangeDecisionData phase_change_decision_data{true, false};
    const auto decision_result = phase_change.arbitrate_phase_change(
        make_not_null(&phase_change_decision_data), Parallel::Phase::Evolve,
        cache);
    CHECK(
        decision_result ==
        std::make_pair(Parallel::Phase::WriteCheckpoint,
                       PhaseControl::ArbitrationStrategy::RunPhaseImmediately));
    CHECK(phase_change_decision_data == PhaseChangeDecisionData{false, true});
  }
  {
    INFO("After WriteCheckpoint, jump to Exit");
    PhaseChangeDecisionData phase_change_decision_data{false, true};
    const auto decision_result = phase_change.arbitrate_phase_change(
        make_not_null(&phase_change_decision_data),
        Parallel::Phase::WriteCheckpoint, cache);
    CHECK(
        decision_result ==
        std::make_pair(Parallel::Phase::Exit,
                       PhaseControl::ArbitrationStrategy::RunPhaseImmediately));
    CHECK(phase_change_decision_data == PhaseChangeDecisionData{false, false});
  }
  {
    INFO("If both decisions are true in Evolve, prioritize WriteCheckpoint");
    PhaseChangeDecisionData phase_change_decision_data{true, true};
    const auto decision_result = phase_change.arbitrate_phase_change(
        make_not_null(&phase_change_decision_data), Parallel::Phase::Evolve,
        cache);
    CHECK(
        decision_result ==
        std::make_pair(Parallel::Phase::WriteCheckpoint,
                       PhaseControl::ArbitrationStrategy::RunPhaseImmediately));
    CHECK(phase_change_decision_data == PhaseChangeDecisionData{false, true});
  }
}
