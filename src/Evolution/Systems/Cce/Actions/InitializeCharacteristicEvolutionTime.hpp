// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Time/ChooseLtsStepSize.hpp"
#include "Time/LtsMode.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/AdaptiveSteppingDiagnostics.hpp"
#include "Time/Tags/LtsMode.hpp"
#include "Time/Tags/StepNumberWithinSlab.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Rational.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

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
 *  - `Tags::AdaptiveSteppingDiagnostics`
 * - Removes: nothing
 */
struct InitializeCharacteristicEvolutionTime {
  using simple_tags_from_options =
      tmpl::list<Initialization::Tags::InitialSlabSize,
                 ::Initialization::Tags::InitialTimeDelta>;

  using const_global_cache_tags = tmpl::list<
      Tags::CceEvolutionPrefix<::Tags::ConcreteTimeStepper<LtsTimeStepper>>,
      Tags::CceEvolutionPrefix<::Tags::LtsModeForced<LtsMode::Conservative>>>;

  using simple_tags =
      tmpl::list<::Tags::TimeStepId, ::Tags::Next<::Tags::TimeStepId>,
                 ::Tags::TimeStep, ::Tags::Time, ::Tags::StepNumberWithinSlab,
                 ::Tags::AdaptiveSteppingDiagnostics>;
  using compute_tags =
      tmpl::transform<time_stepper_ref_tags<LtsTimeStepper>,
                      tmpl::bind<Tags::CceEvolutionPrefix, tmpl::_1>>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const auto& time_stepper =
        db::get<Tags::CceEvolutionPrefix<::Tags::TimeStepper<TimeStepper>>>(
            box);
    const double initial_time_value = db::get<Tags::StartTime>(box);

    double unused_slab_size_goal{};
    db::mutate_apply<
        tmpl::list<::Tags::Next<::Tags::TimeStepId>, ::Tags::TimeStep>,
        tmpl::list<>>(
        Initialization::TimeStepping<Metavariables, TimeStepper, false, true>{},
        make_not_null(&box), make_not_null(&unused_slab_size_goal),
        initial_time_value,
        db::get<::Initialization::Tags::InitialTimeDelta>(box),
        db::get<::Initialization::Tags::InitialSlabSize>(box), time_stepper,
        LtsMode::Conservative);

    Initialization::mutate_assign<tmpl::list<::Tags::TimeStepId, ::Tags::Time>>(
        make_not_null(&box), TimeStepId{}, initial_time_value);
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

}  // namespace Actions
}  // namespace Cce
