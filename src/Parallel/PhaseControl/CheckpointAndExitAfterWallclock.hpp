// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <optional>
#include <pup.h>
#include <string>
#include <type_traits>
#include <utility>

#include "Options/Auto.hpp"
#include "Options/Options.hpp"
#include "Parallel/AlgorithmMetafunctions.hpp"
#include "Parallel/ExitCode.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/ContributeToPhaseChangeReduction.hpp"
#include "Parallel/PhaseControl/PhaseChange.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/System/ParallelInfo.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/UtcTime.hpp"

namespace PhaseControl {

namespace Tags {
/// Storage in the phase change decision tuple so that the Main chare can record
/// the phase to go to when restarting the run from a checkpoint file.
///
/// \note This tag is not intended to participate in any of the reduction
/// procedures, so will error if the combine method is called.
struct RestartPhase {
  using type = std::optional<Parallel::Phase>;

  struct combine_method {
    [[noreturn]] std::optional<Parallel::Phase> operator()(
        const std::optional<Parallel::Phase> /*first_phase*/,
        const std::optional<Parallel::Phase>& /*second_phase*/);
  };

  using main_combine_method = combine_method;
};

/// Stores whether the checkpoint and exit has been requested.
///
/// Combinations are performed via `funcl::Or`, as the phase in question should
/// be chosen if any component requests the jump.
struct CheckpointAndExitRequested {
  using type = bool;

  using combine_method = funcl::Or<>;
  using main_combine_method = funcl::Or<>;
};

}  // namespace Tags

/*!
 * \brief Phase control object that runs the WriteCheckpoint and Exit phases
 * after a specified amount of wallclock time has elapsed.
 *
 * When the executable exits from here, it does so with
 * `Parallel::ExitCode::ContinueFromCheckpoint`.
 *
 * This phase control is useful for running SpECTRE executables performing
 * lengthy computations that may exceed a supercomputer's wallclock limits.
 * Writing a single checkpoint at the end of the job's allocated time allows
 * the computation to be continued, while minimizing the disc space taken up by
 * checkpoint files.
 *
 * When restarting from the checkpoint, this phase control sends the control
 * flow to a UpdateOptionsAtRestartFromCheckpoint phase, allowing the user to
 * update (some) simulation parameters for the continuation of the run.
 *
 * Note that this phase control is not a trigger on wallclock time. Rather,
 * it checks the elapsed wallclock time when called, likely from a global sync
 * point triggered by some other mechanism, e.g., at some slab boundary.
 * Therefore, the WriteCheckpoint and Exit phases will run the first time
 * this phase control is called after the specified wallclock time has been
 * reached.
 *
 * \warning the global sync points _must_ be triggered often enough to ensure
 * there will be at least one sync point (i.e., one call to this phase control)
 * in the window between the requested checkpoint-and-exit time and the time at
 * which the batch system will kill the executable. To make this more concrete,
 * consider this example: when running on a 12-hour queue with a
 * checkpoint-and-exit requested after 11.5 hours, there is a 0.5-hour window
 * for a global sync to occur, the checkpoint files to be written to disc, and
 * the executable to clean up. In this case, triggering a global sync every
 * 2-10 minutes might be desirable. Matching the global sync frequency with the
 * time window for checkpoint and exit is the responsibility of the user!
 *
 * \parblock
 * \warning If modifying the phase-change logic on a
 * checkpoint-restart, this PhaseChange must remain in the list after
 * modification so that the end of the restart logic will run.  The
 * WallclockHours can be changed to None to disable further restarts.
 * \endparblock
 */
struct CheckpointAndExitAfterWallclock : public PhaseChange {
  CheckpointAndExitAfterWallclock(const std::optional<double> wallclock_hours,
                                  const Options::Context& context = {});

  explicit CheckpointAndExitAfterWallclock(CkMigrateMessage* msg);

  /// \cond
  CheckpointAndExitAfterWallclock() = default;
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(CheckpointAndExitAfterWallclock);  // NOLINT
  /// \endcond

  struct WallclockHours {
    using type = Options::Auto<double, Options::AutoLabel::None>;
    static constexpr Options::String help = {
        "Time in hours after which to write the checkpoint and exit. "
        "If 'None' is specified, no action will be taken."};
  };

  using options = tmpl::list<WallclockHours>;
  static constexpr Options::String help{
      "Once the wallclock time has exceeded the specified amount, trigger "
      "writing a checkpoint and then exit with the 'ContinueFromCheckpoint' "
      "exit code."};

  using argument_tags = tmpl::list<>;
  using return_tags = tmpl::list<>;

  using phase_change_tags_and_combines =
      tmpl::list<Tags::RestartPhase, Tags::CheckpointAndExitRequested>;

  template <typename Metavariables>
  using participating_components = typename Metavariables::component_list;

  template <typename... DecisionTags>
  void initialize_phase_data_impl(
      const gsl::not_null<tuples::TaggedTuple<DecisionTags...>*>
          phase_change_decision_data) const;

  template <typename ParallelComponent, typename ArrayIndex,
            typename Metavariables>
  void contribute_phase_data_impl(Parallel::GlobalCache<Metavariables>& cache,
                                  const ArrayIndex& array_index) const;

  template <typename... DecisionTags, typename Metavariables>
  typename std::optional<std::pair<Parallel::Phase, ArbitrationStrategy>>
  arbitrate_phase_change_impl(
      const gsl::not_null<tuples::TaggedTuple<DecisionTags...>*>
          phase_change_decision_data,
      const Parallel::Phase current_phase,
      const Parallel::GlobalCache<Metavariables>& /*cache*/) const;

  void pup(PUP::er& p) override;

 private:
  std::optional<double> wallclock_hours_for_checkpoint_and_exit_ = std::nullopt;
  // This flag is set during arbitration when the class decides to
  // halt the run.  As it is not checkpointed, this distinguishes the
  // state immediately after writing the checkpoint from that
  // immediately after reading it during the restart.
  //
  // Phase arbitration is only run from Main, so there are no
  // threading issues here.
  // NOLINTNEXTLINE(spectre-mutable)
  mutable bool halting_ = false;
};

template <typename... DecisionTags>
void CheckpointAndExitAfterWallclock::initialize_phase_data_impl(
    const gsl::not_null<tuples::TaggedTuple<DecisionTags...>*>
        phase_change_decision_data) const {
  tuples::get<Tags::RestartPhase>(*phase_change_decision_data) = std::nullopt;
  tuples::get<Tags::CheckpointAndExitRequested>(*phase_change_decision_data) =
      false;
}

template <typename ParallelComponent, typename ArrayIndex,
          typename Metavariables>
void CheckpointAndExitAfterWallclock::contribute_phase_data_impl(
    Parallel::GlobalCache<Metavariables>& cache,
    const ArrayIndex& array_index) const {
  if constexpr (std::is_same_v<typename ParallelComponent::chare_type,
                               Parallel::Algorithms::Array>) {
    Parallel::contribute_to_phase_change_reduction<ParallelComponent>(
        tuples::TaggedTuple<Tags::CheckpointAndExitRequested>{true}, cache,
        array_index);
  } else {
    Parallel::contribute_to_phase_change_reduction<ParallelComponent>(
        tuples::TaggedTuple<Tags::CheckpointAndExitRequested>{true}, cache);
  }
}

template <typename... DecisionTags, typename Metavariables>
typename std::optional<std::pair<Parallel::Phase, ArbitrationStrategy>>
CheckpointAndExitAfterWallclock::arbitrate_phase_change_impl(
    const gsl::not_null<tuples::TaggedTuple<DecisionTags...>*>
        phase_change_decision_data,
    const Parallel::Phase current_phase,
    const Parallel::GlobalCache<Metavariables>& /*cache*/) const {
  const double elapsed_hours = sys::wall_time() / 3600.0;

  auto& restart_phase =
      tuples::get<Tags::RestartPhase>(*phase_change_decision_data);
  auto& exit_code =
      tuples::get<Parallel::Tags::ExitCode>(*phase_change_decision_data);
  if (restart_phase.has_value()) {
    // This `if` branch, where restart_phase has a value, is the
    // post-checkpoint call to arbitrate_phase_change.
    if (halting_) {
      // Preserve restart_phase for use after restarting from the checkpoint
      exit_code = Parallel::ExitCode::ContinueFromCheckpoint;
      return std::make_pair(Parallel::Phase::Exit,
                            ArbitrationStrategy::RunPhaseImmediately);
    } else {
      // if current_phase is WriteCheckpoint, we follow with updating options
      if (current_phase == Parallel::Phase::WriteCheckpoint) {
        Parallel::printf("Restarting from checkpoint. Date and time: %s\n",
                         utc_time());
        return std::make_pair(
            Parallel::Phase::UpdateOptionsAtRestartFromCheckpoint,
            ArbitrationStrategy::PermitAdditionalJumps);
      }
      // Reset restart_phase until it is needed for the next checkpoint
      const auto result = restart_phase;
      restart_phase.reset();
      return std::make_pair(result.value(),
                            ArbitrationStrategy::PermitAdditionalJumps);
    }
  }

  auto& checkpoint_and_exit_requested =
      tuples::get<Tags::CheckpointAndExitRequested>(
          *phase_change_decision_data);
  if (checkpoint_and_exit_requested) {
    checkpoint_and_exit_requested = false;
    if (elapsed_hours >= wallclock_hours_for_checkpoint_and_exit_.value_or(
                             std::numeric_limits<double>::infinity())) {
      // Record phase and actual elapsed time for determining following phase
      restart_phase = current_phase;
      ASSERT(not halting_, "Halting for checkpoint recursively");
      halting_ = true;
      return std::make_pair(Parallel::Phase::WriteCheckpoint,
                            ArbitrationStrategy::RunPhaseImmediately);
    }
  }
  return std::nullopt;
}
}  // namespace PhaseControl
