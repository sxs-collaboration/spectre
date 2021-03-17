// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/AlgorithmMetafunctions.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/PhaseControl/PhaseChange.hpp"
#include "Parallel/PhaseControl/PhaseControlTags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Utilities/TaggedTuple.hpp"

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
 *
 * \warning This action should almost always be placed at the end of the action
 * list for a given phase, because the record of the index through the action
 * list will not be retained during a phase change. Therefore, to avoid
 * unexpected behavior, the phase changes should happen at the end of a phase,
 * so that if control returns to the same phase it may continue looping as
 * usual.
 * If this suggestion is ignored, the typical pathology is an infinite loop of
 * repeatedly triggering a phase change, because the trigger is examined too
 * early in the action list for any alteration to have occurred before being
 * interrupted again for a phase change.
 */
template <typename PhaseChangeRegistrars>
struct ExecutePhaseChange {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&, Parallel::AlgorithmExecution> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*component*/) noexcept {
    const auto& phase_change_and_triggers =
        Parallel::get<Tags::PhaseChangeAndTriggers<PhaseChangeRegistrars>>(
            cache);
    bool should_halt = false;
    for (const auto& [trigger, phase_changes] : phase_change_and_triggers) {
      if (trigger->is_triggered(box)) {
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
    return {std::move(box), should_halt
                                ? Parallel::AlgorithmExecution::Halt
                                : Parallel::AlgorithmExecution::Continue};
  }
};
}  // namespace Actions

/*!
 * \brief Use the runtime data aggregated in `phase_change_decision_data` to
 * decide which phase to execute next.
 *
 * \details This function will iterate through each of the option-created pairs
 * of `PhaseChange`s, and obtain from each a
 * `std::optional<std::pair<Metavariables::Phase,
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
template <typename PhaseChangeRegistrars, typename... DecisionTags,
          typename Metavariables>
typename std::optional<typename Metavariables::Phase> arbitrate_phase_change(
    const gsl::not_null<tuples::TaggedTuple<DecisionTags...>*>
        phase_change_decision_data,
    typename Metavariables::Phase current_phase,
    const Parallel::GlobalCache<Metavariables>& cache) noexcept {
  const auto& phase_change_and_triggers =
      Parallel::get<Tags::PhaseChangeAndTriggers<PhaseChangeRegistrars>>(
          cache);
  bool phase_chosen = false;
  for (const auto& [trigger, phase_changes] : phase_change_and_triggers) {
    // avoid unused variable warning
    (void)trigger;
    for (const auto& phase_change : phase_changes) {
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
}

/*!
 * \brief Initialize the Main chare's `phase_change_decision_data` for the
 * option-selected `PhaseChange`s.
 *
 * \details This struct provides a convenient method of specifying the
 * initialization of the `phase_change_decision_data`. To instruct the Main
 * chare to use this initialization routine, define the type alias in the
 * `Metavariables`:
 * ```
 * using initialize_phase_data =
 *   PhaseControl::InitializePhaseChangeDecisionData<phase_change_registrars>;
 * ```
 */
template <typename PhaseChangeRegistrars>
struct InitializePhaseChangeDecisionData {
  template <typename... DecisionTags, typename Metavariables>
  static void apply(
      const gsl::not_null<tuples::TaggedTuple<DecisionTags...>*>
          phase_change_decision_data,
      const Parallel::GlobalCache<Metavariables>& cache) noexcept {
    tuples::get<TagsAndCombines::UsePhaseChangeArbitration>(
        *phase_change_decision_data) = false;
    const auto& phase_change_and_triggers =
        Parallel::get<Tags::PhaseChangeAndTriggers<PhaseChangeRegistrars>>(
            cache);
    for (const auto& [trigger, phase_changes] : phase_change_and_triggers) {
      // avoid unused variable warning
      (void)trigger;
      for (const auto& phase_change : phase_changes) {
        phase_change->initialize_phase_data(phase_change_decision_data);
      }
    }
  }
};
}  // namespace PhaseControl
