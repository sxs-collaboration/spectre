// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Time/Tags.hpp"

namespace Cce {
namespace Actions {

/*!
 * \ingroup ActionsGroup
 * \brief Terminates if the current `::Tags::TimeStepId` has time value later or
 * equal to `Tags::EndTime`.
 *
 * \details Uses:
 * - DataBox:
 *   - `Cce::Tags::EndTime`
 *   - `Tags::TimeStepId`
 *
 * \ref DataBoxGroup changes:
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies: nothing
 *
 */
struct ExitIfEndTimeReached {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    return {db::get<::Tags::TimeStepId>(box).substep_time() >=
                    db::get<Tags::EndTime>(box)
                ? Parallel::AlgorithmExecution::Pause
                : Parallel::AlgorithmExecution::Continue,
            std::nullopt};
  }
};

}  // namespace Actions
}  // namespace Cce
