// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/ContributeToPhaseChangeReduction.hpp"
#include "Parallel/PhaseControl/PhaseChange.hpp"
#include "Parallel/PhaseControl/PhaseControlTags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db
/// \endcond

namespace PhaseControl {
namespace Actions {

/*!
 * \ingroup ActionsGroup
 * \brief Check if any triggers are activated, and perform phase changes as
 * needed.
 *
 * This action is intended to be executed on every component that repeatedly
 * runs iterable actions that would need to halt during a phase change. This
 * action sends data to the Main chare via a reduction.
 *
 * This action iterates over the `Tags::PhaseChangeAndTriggers`, sending
 * reduction data for the phase decision for each triggered `PhaseChange`, then
 * halts the algorithm execution so that the `Main` chare can make a phase
 * decision if any were triggered.
 *
 * Uses:
 * - GlobalCache: `Tags::PhaseChangeAndTriggers`
 * - DataBox: As specified by the `PhaseChange` option-created objects.
 *   - `PhaseChange` objects are permitted to perform mutations on the
 *     \ref DataBoxGroup "DataBox" to store persistent state information.
 */
struct ExecutePhaseChange {
  using const_global_cache_tags =
      tmpl::list<PhaseControl::Tags::PhaseChangeAndTriggers>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*component*/) {
    const auto& phase_change_and_triggers =
        Parallel::get<Tags::PhaseChangeAndTriggers>(cache);
    bool should_halt = false;
    for (const auto& trigger_and_phase_changes : phase_change_and_triggers) {
      const auto& trigger = trigger_and_phase_changes.trigger;
      if (trigger->is_triggered(box)) {
        const auto& phase_changes = trigger_and_phase_changes.phase_changes;
        for (const auto& phase_change : phase_changes) {
          phase_change->template contribute_phase_data<ParallelComponent>(
              make_not_null(&box), cache, array_index);
        }
        should_halt = true;
      }
    }
    // if we halt, we need to make sure that the Main chare knows that it is
    // because we are requesting phase change arbitration, regardless of what
    // data was actually sent to make that decision.
    if (should_halt) {
      if constexpr (std::is_same_v<typename ParallelComponent::chare_type,
                    Parallel::Algorithms::Array>) {
        Parallel::contribute_to_phase_change_reduction<ParallelComponent>(
            tuples::TaggedTuple<TagsAndCombines::UsePhaseChangeArbitration>{
                true},
            cache, array_index);
      } else {
        Parallel::contribute_to_phase_change_reduction<ParallelComponent>(
            tuples::TaggedTuple<TagsAndCombines::UsePhaseChangeArbitration>{
                true},
            cache);
      }
    }
    return {should_halt ? Parallel::AlgorithmExecution::Halt
                        : Parallel::AlgorithmExecution::Continue,
            std::nullopt};
  }
};
}  // namespace Actions

/*!
 * \brief Use the runtime data aggregated in `phase_change_decision_data` to
 * decide which phase to execute next.
 *
 * \details This function will iterate through each of the option-created pairs
 * of `PhaseChange`s, and obtain from each a
 * `std::optional<std::pair<Parallel::Phase,
 * PhaseControl::ArbitrationStrategy>`. Any `std::nullopt` is skipped. If all
 * `PhaseChange`s provide `std::nullopt`, the phase will either keep its
 * current value (if the halt was caused by one of the triggers associated with
 * an  option-created `PhaseChange`), or this function will return a
 * `std::nullopt` as well (otherwise), indicating that the phase should proceed
 * according to other information, such as global ordering.
 *
 * In the case of a `PhaseControl::ArbitrationStrategy::RunPhaseImmediately`,
 * the first such return value is immediately run, and no further `PhaseChange`s
 * are queried for their input.
 *
 * \note There can be cases where multiple triggers activate, and/or multiple
 * `PhaseChange` objects have data in a state for which they would request a
 * specific phase. When multiple phases are requested, arbitration will
 * proceed in order of appearance in the `PhaseChangeAndTriggers`, determined
 * from the input file options. Therefore, if that order of execution is
 * important for the logic of the executable, the input file ordering and
 * `ArbitrationStrategy` must be chosen carefully.
 */
template <typename... DecisionTags, typename Metavariables>
typename std::optional<Parallel::Phase> arbitrate_phase_change(
    const gsl::not_null<tuples::TaggedTuple<DecisionTags...>*>
        phase_change_decision_data,
    Parallel::Phase current_phase,
    const Parallel::GlobalCache<Metavariables>& cache) {
  if constexpr (tmpl::list_contains_v<
                    typename Parallel::GlobalCache<Metavariables>::tags_list,
                    Tags::PhaseChangeAndTriggers>) {
    const auto& phase_change_and_triggers =
        Parallel::get<Tags::PhaseChangeAndTriggers>(cache);
    bool phase_chosen = false;
    for (const auto& trigger_and_phase_changes : phase_change_and_triggers) {
      for (const auto& phase_change : trigger_and_phase_changes.phase_changes) {
        const auto phase_result = phase_change->arbitrate_phase_change(
            phase_change_decision_data, current_phase, cache);
        if (phase_result.has_value()) {
          if (phase_result.value().second ==
              ArbitrationStrategy::RunPhaseImmediately) {
            tuples::get<TagsAndCombines::UsePhaseChangeArbitration>(
                *phase_change_decision_data) = false;
            return phase_result.value().first;
          }
          current_phase = phase_result.value().first;
          phase_chosen = true;
        }
      }
    }
    if (tuples::get<TagsAndCombines::UsePhaseChangeArbitration>(
            *phase_change_decision_data) == false and
        not phase_chosen) {
      return std::nullopt;
    }
    // if no phase change object suggests a specific phase, return to execution
    // in the current phase.
    tuples::get<TagsAndCombines::UsePhaseChangeArbitration>(
        *phase_change_decision_data) = false;
    return current_phase;
  } else {
    (void)phase_change_decision_data;
    (void)current_phase;
    (void)cache;
    return std::nullopt;
  }
}
}  // namespace PhaseControl
