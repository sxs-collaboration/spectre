// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/ReceiveTags.hpp"
#include "Parallel/Actions/Receive.hpp"
#include "Parallel/ExtractFromInbox.hpp"
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
struct ReceiveWorldtubeData
    : Parallel::Actions::Receive<
          Cce::ReceiveTags::BoundaryData<
              typename Metavariables::cce_boundary_communication_tags>,
          ::Tags::TimeStepId> {
  template <
      typename DbTags, typename... InboxTags, typename ArrayIndex,
      typename ActionList, typename ParallelComponent,
      Requires<
          tmpl::list_contains_v<
              tmpl::list<InboxTags...>,
              Cce::ReceiveTags::BoundaryData<
                  typename Metavariables::cce_boundary_communication_tags>> and
          tmpl::list_contains_v<DbTags, ::Tags::TimeStepId>> = nullptr>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& inboxes,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const auto received_boundary_data = Parallel::extract_from_inbox<
        Cce::ReceiveTags::BoundaryData<
            typename Metavariables::cce_boundary_communication_tags>,
        ::Tags::TimeStepId>(inboxes, box);
    tmpl::for_each<typename Metavariables::cce_boundary_communication_tags>(
        [&received_boundary_data, &box](auto tag_v) noexcept {
          using tag = typename decltype(tag_v)::type;
          db::mutate<tag>(
              make_not_null(&box),
              [&received_boundary_data](const gsl::not_null<typename tag::type*>
                                            destination) noexcept {
                *destination = get<tag>(received_boundary_data);
              });
        });
    return std::forward_as_tuple(std::move(box));
  }
  template <
      typename DbTags, typename... InboxTags, typename ArrayIndex,
      typename ActionList, typename ParallelComponent,
      Requires<
          not tmpl::list_contains_v<
              tmpl::list<InboxTags...>,
              Cce::ReceiveTags::BoundaryData<
                  typename Metavariables::cce_boundary_communication_tags>> or
          not tmpl::list_contains_v<DbTags, ::Tags::TimeStepId>> = nullptr>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    ERROR(
        "Required tags not present in the inbox or databox to transfer the CCE "
        "boundary data");
    // provided for return type inference. This line will never be executed.
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace Cce
