// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}  // namespace Parallel
/// \endcond

namespace Parallel::Actions {

/*!
 * \brief Wait for the `InboxTag` to be received at the current temporal ID
 *
 * This class does not provide an implementation of `apply`. Instead, actions
 * can derive from this class to inherit the `is_ready` function and implement
 * their own `apply` function. The `Parallel::extract_from_inbox` function can
 * be useful to implement the `apply` function. Here's an example for an action
 * that derives from this class:
 *
 * \snippet Parallel/Actions/Test_Receive.cpp receive_action
 */
template <typename InboxTag, typename TemporalIdTag>
struct Receive {
  using inbox_tags = tmpl::list<InboxTag>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex>
  static bool is_ready(const db::DataBox<DbTags>& box,
                       const tuples::TaggedTuple<InboxTags...>& inboxes,
                       const Parallel::GlobalCache<Metavariables>& /*cache*/,
                       const ArrayIndex& /*array_index*/) noexcept {
    const auto& inbox = get<InboxTag>(inboxes);
    return inbox.find(db::get<TemporalIdTag>(box)) != inbox.end();
  }
};

}  // namespace Parallel::Actions
