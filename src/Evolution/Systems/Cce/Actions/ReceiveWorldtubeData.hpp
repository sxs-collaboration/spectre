// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/ReceiveTags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {
namespace Actions {

/*!
 * \ingroup ActionsGroup
 * \brief Takes the boundary data needed to perform the CCE linear solves as
 * arguments and puts them in the \ref DataBoxGroup, updating the
 * `Cce::Tags::BoundaryTime` accordingly.
 *
 * \details The boundary data is computed by a separate component, and packaged
 * into a `Variables<tmpl::list<BoundaryTags...>>` which is sent in the argument
 * of the simple action invocation. The `TimeStepId` is also provided to confirm
 * the time associated with the passed boundary data.
 *
 * \ref DataBoxGroup changes:
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies:
 *   - All tags in `BoundaryTags`
 *   - `Cce::Tags::BoundaryTime`
 */
template <typename Metavariables>
struct ReceiveWorldtubeData {
  using inbox_tags = tmpl::list<Cce::ReceiveTags::BoundaryData<
      typename Metavariables::cce_boundary_communication_tags>>;

  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    auto& inbox = tuples::get<Cce::ReceiveTags::BoundaryData<
        typename Metavariables::cce_boundary_communication_tags>>(inboxes);
    if (inbox.count(db::get<::Tags::TimeStepId>(box)) != 1) {
      return {Parallel::AlgorithmExecution::Pause,
              tmpl::index_of<ActionList, ReceiveWorldtubeData>::value};
    }

    tmpl::for_each<typename Metavariables::cce_boundary_communication_tags>(
        [&inbox, &box](auto tag_v) {
          using tag = typename decltype(tag_v)::type;
          db::mutate<tag>(
              [&inbox](const gsl::not_null<typename tag::type*> destination,
                       const TimeStepId& time) {
                *destination = get<tag>(inbox[time]);
              },
              make_not_null(&box), db::get<::Tags::TimeStepId>(box));
        });
    inbox.erase(db::get<::Tags::TimeStepId>(box));
    return {Parallel::AlgorithmExecution::Continue,
            tmpl::index_of<ActionList, ReceiveWorldtubeData>::value + 1};
  }
};
}  // namespace Actions
}  // namespace Cce
