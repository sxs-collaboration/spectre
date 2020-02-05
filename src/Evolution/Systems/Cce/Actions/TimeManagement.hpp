// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeStepId.hpp"

namespace Cce {
namespace Actions {

/*!
 * \ingroup ActionGroup
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
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::tuple<db::DataBox<DbTags>&&, bool>(
        std::move(box),
        db::get<::Tags::TimeStepId>(box).substep_time().value() >=
            db::get<Tags::EndTime>(box));
  }
};

}  // namespace Actions
}  // namespace Cce
