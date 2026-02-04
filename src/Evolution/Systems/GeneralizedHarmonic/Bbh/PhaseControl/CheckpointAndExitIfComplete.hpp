// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <pup.h>
#include <string>
#include <type_traits>
#include <utility>

#include "Evolution/Systems/GeneralizedHarmonic/Bbh/CompletionCriteria.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/ContributeToPhaseChangeReduction.hpp"
#include "Parallel/PhaseControl/PhaseChange.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace gh::bbh::phase_control {
namespace Tags {
/// Storage in the phase-change decision tuple for whether the
/// BBH completion path has requested completion.
struct CheckpointRequested {
  using type = bool;
  using combine_method = funcl::Or<>;
  using main_combine_method = funcl::Or<>;
};

/// Storage in the phase-change decision tuple for jumping from
/// `WriteCheckpoint` to `Exit`.
struct ExitAfterWriteCheckpoint {
  using type = bool;

  struct combine_method {
    bool operator()(const bool /*first*/, const bool /*second*/) {
      ERROR(
          "ExitAfterWriteCheckpoint should only be modified during "
          "phase-change arbitration on Main.");
    }
  };

  using main_combine_method = combine_method;
};
}  // namespace Tags

/*!
 * \brief If BBH completion has been requested, jump from `Evolve` to
 * `WriteCheckpoint`, then immediately to `Exit`.
 *
 * \details The `WriteCheckpoint` phase executes checkpoint actions, typically
 * including final volume data writes, as specified by
 * `EventsAndTriggersAtCheckpoints`.
 */
struct CheckpointAndExitIfComplete : public PhaseChange {
  /// \cond
  CheckpointAndExitIfComplete() = default;
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(CheckpointAndExitIfComplete);  // NOLINT
  /// \endcond

  static std::string name() { return "BbhCheckpointAndExitIfComplete"; }
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "When BBH completion has been requested, jump from Evolve to "
      "WriteCheckpoint so all elements write final volume data at a "
      "synchronized time, then jump to Exit."};

  using argument_tags = tmpl::list<gh::bbh::Tags::ElementCompletionRequested>;
  using return_tags = tmpl::list<>;

  using phase_change_tags_and_combines =
      tmpl::list<Tags::CheckpointRequested, Tags::ExitAfterWriteCheckpoint>;

  template <typename Metavariables>
  using participating_components = typename Metavariables::component_list;

  template <typename... DecisionTags>
  void initialize_phase_data_impl(
      const gsl::not_null<tuples::TaggedTuple<DecisionTags...>*>
          phase_change_decision_data) const {
    tuples::get<Tags::CheckpointRequested>(*phase_change_decision_data) = false;
    tuples::get<Tags::ExitAfterWriteCheckpoint>(*phase_change_decision_data) =
        false;
  }

  template <typename ParallelComponent, typename ArrayIndex,
            typename Metavariables>
  void contribute_phase_data_impl(const bool element_completion_requested,
                                  Parallel::GlobalCache<Metavariables>& cache,
                                  const ArrayIndex& array_index) const {
    if constexpr (std::is_same_v<typename ParallelComponent::chare_type,
                                 Parallel::Algorithms::Array>) {
      Parallel::contribute_to_phase_change_reduction<ParallelComponent>(
          tuples::TaggedTuple<Tags::CheckpointRequested>{
              element_completion_requested},
          cache, array_index);
    } else {
      Parallel::contribute_to_phase_change_reduction<ParallelComponent>(
          tuples::TaggedTuple<Tags::CheckpointRequested>{
              element_completion_requested},
          cache);
    }
  }

  template <typename... DecisionTags, typename Metavariables>
  std::optional<std::pair<Parallel::Phase, PhaseControl::ArbitrationStrategy>>
  arbitrate_phase_change_impl(
      const gsl::not_null<tuples::TaggedTuple<DecisionTags...>*>
          phase_change_decision_data,
      const Parallel::Phase current_phase,
      const Parallel::GlobalCache<Metavariables>& /*cache*/) const {
    auto& checkpoint_requested =
        tuples::get<Tags::CheckpointRequested>(*phase_change_decision_data);
    auto& exit_after_write_checkpoint =
        tuples::get<Tags::ExitAfterWriteCheckpoint>(
            *phase_change_decision_data);

    if (current_phase == Parallel::Phase::WriteCheckpoint and
        exit_after_write_checkpoint) {
      exit_after_write_checkpoint = false;
      return std::make_pair(
          Parallel::Phase::Exit,
          PhaseControl::ArbitrationStrategy::RunPhaseImmediately);
    }

    if (current_phase == Parallel::Phase::Evolve and checkpoint_requested) {
      checkpoint_requested = false;
      exit_after_write_checkpoint = true;
      return std::make_pair(
          Parallel::Phase::WriteCheckpoint,
          PhaseControl::ArbitrationStrategy::RunPhaseImmediately);
    }

    checkpoint_requested = false;
    return std::nullopt;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;
};
}  // namespace gh::bbh::phase_control
