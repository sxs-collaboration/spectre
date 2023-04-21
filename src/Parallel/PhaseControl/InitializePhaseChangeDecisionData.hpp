// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Parallel/GlobalCache.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseControl/PhaseControlTags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace PhaseControl {
/*!
 * \brief Initialize the Main chare's `phase_change_decision_data` for the
 * option-selected `PhaseChange`s.
 */
template <typename... DecisionTags, typename Metavariables>
void initialize_phase_change_decision_data(
    const gsl::not_null<tuples::TaggedTuple<DecisionTags...>*>
        phase_change_decision_data,
    const Parallel::GlobalCache<Metavariables>& cache) {
  tuples::get<TagsAndCombines::UsePhaseChangeArbitration>(
      *phase_change_decision_data) = false;
  if constexpr (Parallel::is_in_global_cache<Metavariables,
                                             Tags::PhaseChangeAndTriggers>) {
    const auto& phase_change_and_triggers =
        Parallel::get<Tags::PhaseChangeAndTriggers>(cache);
    for (const auto& trigger_and_phase_changes : phase_change_and_triggers) {
      for (const auto& phase_change : trigger_and_phase_changes.phase_changes) {
        phase_change->template initialize_phase_data<Metavariables>(
            phase_change_decision_data);
      }
    }
  }
}
}  // namespace PhaseControl
