// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <pup.h>
#include <string>
#include <type_traits>
#include <utility>

#include "Options/Auto.hpp"
#include "Options/Options.hpp"
#include "Parallel/AlgorithmMetafunctions.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/PhaseControl/PhaseChange.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/System/ParallelInfo.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PhaseControl {
template <typename Metavariables, typename PhaseChangeRegistrars>
class CheckpointAndExitAfterWallclock;

namespace Registrars {
template <typename Metavariables>
struct CheckpointAndExitAfterWallclock {
  template <typename PhaseChangeRegistrars>
  using f =
      ::PhaseControl::CheckpointAndExitAfterWallclock<Metavariables,
                                                      PhaseChangeRegistrars>;
};
}  // namespace Registrars
/// \endcond

namespace Tags {
/// Storage in the phase change decision tuple so that the Main chare can record
/// the phase to go to when restarting the run from a checkpoint file.
///
/// \note This tag is not intended to participate in any of the reduction
/// procedures, so will error if the combine method is called.
template <typename PhaseType>
struct RestartPhase {
  using type = std::optional<PhaseType>;

  struct combine_method {
    std::optional<PhaseType> operator()(
        const std::optional<PhaseType> /*first_phase*/,
        const std::optional<PhaseType>& /*second_phase*/) noexcept {
      ERROR(
          "The restart phase should only be altered by the phase change "
          "arbitration in the Main chare, so no reduction data should be "
          "provided.");
    }
  };

  using main_combine_method = combine_method;
};

/// Storage in the phase change decision tuple so that the Main chare can record
/// the elapsed wallclock time since the start of the run.
///
/// \note This tag is not intended to participate in any of the reduction
/// procedures, so will error if the combine method is called.
struct WallclockHoursAtCheckpoint {
  using type = std::optional<double>;

  struct combine_method {
    std::optional<double> operator()(
        const std::optional<double> /*first_time*/,
        const std::optional<double>& /*second_time*/) noexcept {
      ERROR(
          "The wallclock time at which a checkpoint was requested should "
          "only be altered by the phase change arbitration in the Main "
          "chare, so no reduction data should be provided.");
    }
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
 * This phase control is useful for running SpECTRE executables performing
 * lengthy computations that may exceed a supercomputer's wallclock limits.
 * Writing a single checkpoint at the end of the job's allocated time allows
 * the computation to be continued, while minimizing the disc space taken up by
 * checkpoint files.
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
 */
template <typename Metavariables,
          typename PhaseChangeRegistrars = tmpl::list<
              Registrars::CheckpointAndExitAfterWallclock<Metavariables>>>
struct CheckpointAndExitAfterWallclock
    : public PhaseChange<PhaseChangeRegistrars> {
  // This PhaseChange only makes sense if Metavars has a WriteCheckpoint phase
  static_assert(Parallel::Algorithm_detail::has_WriteCheckpoint_v<
                    typename Metavariables::Phase>,
                "Requested to write checkpoints but Metavariables::Phase "
                "doesn't have a WriteCheckpoint phase");

  CheckpointAndExitAfterWallclock(const std::optional<double> wallclock_hours,
                                  const Options::Context& context = {})
      : wallclock_hours_for_checkpoint_and_exit_(wallclock_hours) {
    if (wallclock_hours.has_value() and wallclock_hours.value() < 0.0) {
      PARSE_ERROR(context, "Must give a positive time in hours, but got "
                               << wallclock_hours.value());
    }
  }
  explicit CheckpointAndExitAfterWallclock(CkMigrateMessage* msg) noexcept
      : PhaseChange<PhaseChangeRegistrars>(msg) {}

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
      "writing a checkpoint and then exit."};

  using argument_tags = tmpl::list<>;
  using return_tags = tmpl::list<>;

  using phase_change_tags_and_combines =
      tmpl::list<Tags::RestartPhase<typename Metavariables::Phase>,
                 Tags::WallclockHoursAtCheckpoint,
                 Tags::CheckpointAndExitRequested>;

  template <typename LocalMetavariables>
  using participating_components = typename LocalMetavariables::component_list;

  template <typename... DecisionTags>
  void initialize_phase_data_impl(
      const gsl::not_null<tuples::TaggedTuple<DecisionTags...>*>
          phase_change_decision_data) const noexcept {
    tuples::get<Tags::RestartPhase<typename Metavariables::Phase>>(
        *phase_change_decision_data) = std::nullopt;
    tuples::get<Tags::WallclockHoursAtCheckpoint>(*phase_change_decision_data) =
        std::nullopt;
    tuples::get<Tags::CheckpointAndExitRequested>(*phase_change_decision_data) =
        false;
  }

  template <typename ParallelComponent, typename ArrayIndex,
            typename LocalMetavariables>
  void contribute_phase_data_impl(
      Parallel::GlobalCache<LocalMetavariables>& cache,
      const ArrayIndex& array_index) const noexcept {
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

  template <typename... DecisionTags, typename LocalMetavariables>
  typename std::optional<
      std::pair<typename Metavariables::Phase, ArbitrationStrategy>>
  arbitrate_phase_change_impl(
      const gsl::not_null<tuples::TaggedTuple<DecisionTags...>*>
          phase_change_decision_data,
      const typename LocalMetavariables::Phase current_phase,
      const Parallel::GlobalCache<LocalMetavariables>& /*cache*/)
      const noexcept {
    // If no checkpoint-and-exit time given, then do nothing
    if (not wallclock_hours_for_checkpoint_and_exit_.has_value()) {
      return std::nullopt;
    }

    const double elapsed_hours = sys::wall_time() / 3600.0;

    auto& restart_phase =
        tuples::get<Tags::RestartPhase<typename Metavariables::Phase>>(
            *phase_change_decision_data);
    auto& wallclock_hours_at_checkpoint =
        tuples::get<Tags::WallclockHoursAtCheckpoint>(
            *phase_change_decision_data);
    if (restart_phase.has_value()) {
      ASSERT(wallclock_hours_at_checkpoint.has_value(),
             "Consistency error: Should have recorded the Wallclock time "
             "while recording a phase to restart from.");
      // This `if` branch, where restart_phase has a value, is the
      // post-checkpoint call to arbitrate_phase_change. Depending on the time
      // elapsed so far in this run, next phase is...
      // - Exit, if the time is large
      // - restart_phase, if the time is small
      if (elapsed_hours >= wallclock_hours_at_checkpoint.value()) {
        // Preserve restart_phase for use after restarting from the checkpoint
        return std::make_pair(Metavariables::Phase::Exit,
                              ArbitrationStrategy::RunPhaseImmediately);
      } else {
        // Reset restart_phase until it is needed for the next checkpoint
        const auto result = restart_phase;
        restart_phase.reset();
        wallclock_hours_at_checkpoint.reset();
        return std::make_pair(result.value(),
                              ArbitrationStrategy::PermitAdditionalJumps);
      }
    }

    auto& checkpoint_and_exit_requested =
        tuples::get<Tags::CheckpointAndExitRequested>(
            *phase_change_decision_data);
    if (checkpoint_and_exit_requested) {
      checkpoint_and_exit_requested = false;
      // We checked wallclock_hours_for_checkpoint_and_exit_ has value above
      if (elapsed_hours >= wallclock_hours_for_checkpoint_and_exit_.value()) {
        // Record phase and actual elapsed time for determining following phase
        restart_phase = current_phase;
        wallclock_hours_at_checkpoint = elapsed_hours;
        return std::make_pair(Metavariables::Phase::WriteCheckpoint,
                              ArbitrationStrategy::RunPhaseImmediately);
      }
    }
    return std::nullopt;
  }

  void pup(PUP::er& p) noexcept override {
    PhaseChange<PhaseChangeRegistrars>::pup(p);
    p | wallclock_hours_for_checkpoint_and_exit_;
  }

 private:
  std::optional<double> wallclock_hours_for_checkpoint_and_exit_ = std::nullopt;
};
}  // namespace PhaseControl

/// \cond
template <typename Metavariables, typename PhaseChangeRegistrars>
PUP::able::PUP_ID PhaseControl::CheckpointAndExitAfterWallclock<
    Metavariables, PhaseChangeRegistrars>::my_PUP_ID = 0;
/// \endcond
