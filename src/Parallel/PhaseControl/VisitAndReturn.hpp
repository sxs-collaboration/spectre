// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <pup.h>
#include <string>
#include <utility>
#include <type_traits>

#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/ContributeToPhaseChangeReduction.hpp"
#include "Parallel/PhaseControl/PhaseChange.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace PhaseControl {
namespace Tags {
/// Storage in the phase change decision tuple so that the Main chare can record
/// the phase to return to after a temporary phase.
///
/// \note This tag is not intended to participate in any of the reduction
/// procedures, so will error if the combine method is called.
template <Parallel::Phase Phase>
struct ReturnPhase {
  using type = std::optional<Parallel::Phase>;

  struct combine_method {
    std::optional<Parallel::Phase> operator()(
        const std::optional<Parallel::Phase> /*first_phase*/,
        const std::optional<Parallel::Phase>& /*second_phase*/) {
      ERROR(
          "The return phase should only be altered by the phase change "
          "arbitration in the Main chare, so no reduction data should be "
          "provided.");
    }
  };

  using main_combine_method = combine_method;
};

/// Stores whether the phase in question has been requested.
///
/// Combinations are performed via `funcl::Or`, as the phase in question should
/// be chosen if any component requests the jump.
template <Parallel::Phase Phase>
struct TemporaryPhaseRequested {
  using type = bool;

  using combine_method = funcl::Or<>;
  using main_combine_method = funcl::Or<>;
};
}  // namespace Tags

/*!
 * \brief Phase control object for temporarily visiting `TargetPhase`, until the
 * algorithm halts again, then returning to the original phase.
 *
 * The motivation for this type of procedure is e.g. load balancing,
 * checkpointing, and other maintenance tasks that should be performed
 * periodically during a lengthy evolution.
 * Once triggered, this will cause a change to `TargetPhase`, but store the
 * current phase to resume execution when the tasks in `TargetPhase` are
 * completed.
 *
 * Any parallel component can participate in the associated phase change
 * reduction data contribution, and if any component requests the temporary
 * phase, it will execute.
 *
 * \note  If multiple such methods are specified (with different
 * `TargetPhase`s), then the order of phase jumps depends on their order in the
 * list.
 * - If multiple `VisitAndReturn`s trigger simultaneously, then they will visit
 *   in sequence specified by the input file: first going to the first
 *   `TargetPhase` until that phase resolves, then immediately entering the
 *   second `TargetPhase` (without yet returning to the original phase), then
 *   finally returning to the original phase.
 * - If a `VisitAndReturn` is triggered in a phase that is already a
 *   `TargetPhase` of another `VisitAndReturn`, it will be executed, and
 *   following completion, control will return to the original phase from before
 *   the first `VisitAndReturn`.
 */
template <Parallel::Phase TargetPhase>
struct VisitAndReturn : public PhaseChange {
  /// \cond
  VisitAndReturn() = default;
  explicit VisitAndReturn(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(VisitAndReturn);  // NOLINT
  /// \endcond

  static std::string name() {
    return MakeString{} << "VisitAndReturn(" << TargetPhase << ")";
  }
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Temporarily jump to the phase given by `TargetPhase`, returning to the "
      "previously executing phase when complete."};

  using argument_tags = tmpl::list<>;
  using return_tags = tmpl::list<>;

  using phase_change_tags_and_combines =
      tmpl::list<Tags::ReturnPhase<TargetPhase>,
                 Tags::TemporaryPhaseRequested<TargetPhase>>;

  template <typename Metavariables>
  using participating_components = typename Metavariables::component_list;

  template <typename... DecisionTags>
  void initialize_phase_data_impl(
      const gsl::not_null<tuples::TaggedTuple<DecisionTags...>*>
          phase_change_decision_data) const {
    tuples::get<Tags::ReturnPhase<TargetPhase>>(*phase_change_decision_data) =
        std::nullopt;
    tuples::get<Tags::TemporaryPhaseRequested<TargetPhase>>(
        *phase_change_decision_data) = false;
  }

  template <typename ParallelComponent, typename ArrayIndex,
            typename Metavariables>
  void contribute_phase_data_impl(Parallel::GlobalCache<Metavariables>& cache,
                                  const ArrayIndex& array_index) const {
    if constexpr (std::is_same_v<typename ParallelComponent::chare_type,
                                 Parallel::Algorithms::Array>) {
      Parallel::contribute_to_phase_change_reduction<ParallelComponent>(
          tuples::TaggedTuple<Tags::TemporaryPhaseRequested<TargetPhase>>{true},
          cache, array_index);
    } else {
      Parallel::contribute_to_phase_change_reduction<ParallelComponent>(
          tuples::TaggedTuple<Tags::TemporaryPhaseRequested<TargetPhase>>{true},
          cache);
    }
  }

  template <typename... DecisionTags, typename Metavariables>
  typename std::optional<std::pair<Parallel::Phase, ArbitrationStrategy>>
  arbitrate_phase_change_impl(
      const gsl::not_null<tuples::TaggedTuple<DecisionTags...>*>
          phase_change_decision_data,
      const Parallel::Phase current_phase,
      const Parallel::GlobalCache<Metavariables>& /*cache*/) const {
    auto& return_phase = tuples::get<Tags::ReturnPhase<TargetPhase>>(
        *phase_change_decision_data);
    if (return_phase.has_value()) {
      const auto result = return_phase;
      return_phase.reset();
      return std::make_pair(result.value(),
                            ArbitrationStrategy::PermitAdditionalJumps);
    }
    auto& temporary_phase_requested =
        tuples::get<Tags::TemporaryPhaseRequested<TargetPhase>>(
            *phase_change_decision_data);
    if (temporary_phase_requested) {
      return_phase = current_phase;
      temporary_phase_requested = false;
      return std::make_pair(TargetPhase,
                            ArbitrationStrategy::RunPhaseImmediately);
    }
    return std::nullopt;
  }

  void pup(PUP::er& /*p*/) override {}
};
}  // namespace PhaseControl

/// \cond
template <Parallel::Phase TargetPhase>
PUP::able::PUP_ID PhaseControl::VisitAndReturn<TargetPhase>::my_PUP_ID = 0;
/// \endcond
