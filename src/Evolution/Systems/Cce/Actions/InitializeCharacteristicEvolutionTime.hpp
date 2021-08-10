// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Time/StepChoosers/ErrorControl.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Rational.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Cce {
namespace Actions {

/*!
 * \ingroup ActionsGroup
 * \brief Initializes the contents of the `CharacteristicEvolution` component
 * for performing the time evolution of the system, which is the singleton that
 * handles the main evolution system for CCE computations.
 *
 * \details Sets up the \ref DataBoxGroup to be ready to perform the
 * time-stepping associated with the CCE system.
 *
 * \ref DataBoxGroup changes:
 * - Modifies: nothing
 * - Adds:
 *  - `Tags::TimeStepId`
 *  - `Tags::Next<Tags::TimeStepId>`
 *  - `Tags::TimeStep`
 *  - `Tags::Time`
 *  -
 * ```
 * Tags::HistoryEvolvedVariables<
 * metavariables::evolved_coordinates_variables_tag,
 * db::add_tag_prefix<Tags::dt,
 * metavariables::evolved_coordinates_variables_tag>>
 * ```
 *  -
 * ```
 * Tags::HistoryEvolvedVariables<
 * ::Tags::Variables<metavariables::evolved_swsh_tag>,
 * ::Tags::Variables<metavariables::evolved_swsh_dt_tag>>
 * ```
 * - Removes: nothing
 *
 * \note This action relies on the `SetupDataBox` aggregated initialization
 * mechanism, so `Actions::SetupDataBox` must be present in the `Initialization`
 * phase action list prior to this action.
 */
template <typename EvolvedCoordinatesVariablesTag, typename EvolvedSwshTag,
          bool local_time_stepping>
struct InitializeCharacteristicEvolutionTime {
  using initialization_tags_to_keep = tmpl::flatten<tmpl::list<
      Initialization::Tags::InitialSlabSize<local_time_stepping>,
      ::Tags::IsUsingTimeSteppingErrorControl<OptionTags::CceEvolutionPrefix>,
      Tags::CceEvolutionPrefix<::Tags::TimeStepper<LtsTimeStepper>>,
      Tags::CceEvolutionPrefix<::Tags::StepChoosers>,
      Tags::CceEvolutionPrefix<::Tags::StepController>,
      ::Initialization::Tags::InitialTimeDelta>>;

  using initialization_tags = initialization_tags_to_keep;

  using const_global_cache_tags = tmpl::list<>;

  using evolved_swsh_variables_tag =
      ::Tags::Variables<tmpl::list<EvolvedSwshTag>>;
  using simple_tags = tmpl::list<
      ::Tags::TimeStepId, ::Tags::Next<::Tags::TimeStepId>, ::Tags::TimeStep,
      ::Tags::Next<::Tags::TimeStep>, ::Tags::Time,
      ::Tags::HistoryEvolvedVariables<EvolvedCoordinatesVariablesTag>,
      ::Tags::HistoryEvolvedVariables<evolved_swsh_variables_tag>,
      ::Tags::StepperErrorUpdated>;
  using compute_tags = tmpl::list<::Tags::SubstepTimeCompute>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const double initial_time_value = db::get<Tags::StartTime>(box);
    const double slab_size =
        db::get<::Initialization::Tags::InitialSlabSize<local_time_stepping>>(
            box);

    const Slab single_step_slab{initial_time_value,
                                initial_time_value + slab_size};
    const Time initial_time = single_step_slab.start();
    TimeDelta initial_time_step;
    const double initial_time_delta =
        db::get<Initialization::Tags::InitialTimeDelta>(box);
    if constexpr (local_time_stepping) {
      initial_time_step =
          db::get<Tags::CceEvolutionPrefix<::Tags::StepController>>(box)
              .choose_step(initial_time, initial_time_delta);
    } else {
      (void)initial_time_delta;
      initial_time_step = initial_time.slab().duration();
    }

    const auto& time_stepper = db::get<::Tags::TimeStepper<>>(box);

    const size_t starting_order =
        time_stepper.number_of_past_steps() == 0 ? time_stepper.order() : 1;

    typename ::Tags::HistoryEvolvedVariables<EvolvedCoordinatesVariablesTag>::
        type coordinate_history(starting_order);

    typename ::Tags::HistoryEvolvedVariables<evolved_swsh_variables_tag>::type
        swsh_history(starting_order);
    Initialization::mutate_assign<simple_tags>(
        make_not_null(&box), TimeStepId{},
        TimeStepId{true,
                   -static_cast<int64_t>(time_stepper.number_of_past_steps()),
                   initial_time},
        initial_time_step, initial_time_step, initial_time_value,
        std::move(coordinate_history), std::move(swsh_history), false);
    return std::make_tuple(std::move(box));
  }
};

}  // namespace Actions
}  // namespace Cce
