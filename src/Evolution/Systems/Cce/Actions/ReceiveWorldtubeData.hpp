// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/ReceiveTags.hpp"
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
    auto& inbox = tuples::get<Cce::ReceiveTags::BoundaryData<
        typename Metavariables::cce_boundary_communication_tags>>(inboxes);
    tmpl::for_each<typename Metavariables::cce_boundary_communication_tags>(
        [&inbox, &box ](auto tag_v) noexcept {
          using tag = typename decltype(tag_v)::type;
          db::mutate<tag>(
              make_not_null(&box),
              [&inbox](const gsl::not_null<db::item_type<tag>*> destination,
                       const TimeStepId& time) noexcept {
                *destination = get<tag>(inbox[time]);
              },
              db::get<::Tags::TimeStepId>(box));
        });
    inbox.erase(db::get<::Tags::TimeStepId>(box));
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

  template <
      typename DbTags, typename... InboxTags, typename ArrayIndex,
      Requires<
          tmpl::list_contains_v<
              tmpl::list<InboxTags...>,
              Cce::ReceiveTags::BoundaryData<
                  typename Metavariables::cce_boundary_communication_tags>> and
          tmpl::list_contains_v<DbTags, ::Tags::TimeStepId>> = nullptr>
  static bool is_ready(
      const db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/) noexcept {
    return tuples::get<Cce::ReceiveTags::BoundaryData<
               typename Metavariables::cce_boundary_communication_tags>>(
               inboxes)
               .count(db::get<::Tags::TimeStepId>(box)) == 1;
  }

  template <
      typename DbTags, typename... InboxTags, typename ArrayIndex,
      Requires<
          not tmpl::list_contains_v<
              tmpl::list<InboxTags...>,
              Cce::ReceiveTags::BoundaryData<
                  typename Metavariables::cce_boundary_communication_tags>> or
          not tmpl::list_contains_v<DbTags, ::Tags::TimeStepId>> = nullptr>
  static bool is_ready(
      const db::DataBox<DbTags>& /*box*/,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/) noexcept {
    return false;
  }
};
}  // namespace Actions
}  // namespace Cce
