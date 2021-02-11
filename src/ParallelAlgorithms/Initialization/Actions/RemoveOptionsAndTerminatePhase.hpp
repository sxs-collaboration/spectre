// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Initialization {

/// \ingroup InitializationGroup
/// Available actions to be used in the initialization.
///
/// The action list for initialization must end with the
/// `Initialization::Actions::RemoveOptionsAndTerminatePhase` action. For
/// example,
///
/// \snippet Test_RemoveOptionsAndTerminatePhase.cpp actions
///
namespace Actions {
/// \ingroup InitializationGroup
struct RemoveOptionsAndTerminatePhase {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ArrayIndex,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using initialization_tags = Parallel::get_initialization_tags<ActionList>;
    using initialization_tags_to_keep =
        Parallel::get_initialization_tags_to_keep<ActionList>;
    // Keep the tags that are in initialization_tags_to_keep.
    using tags_to_remove = tmpl::remove_if<
        initialization_tags,
        tmpl::bind<tmpl::list_contains, tmpl::pin<initialization_tags_to_keep>,
                   tmpl::_1>>;
    // Retrieve the `initialization_tags` that are still in the DataBox
    // and remove them.
    using tags_to_remove_this_time = tmpl::filter<
        tags_to_remove,
        tmpl::bind<
            tmpl::list_contains,
            tmpl::pin<typename db::DataBox<DbTagsList>::mutable_item_tags>,
            tmpl::_1>>;
    static_assert(std::is_same<tmpl::back<ActionList>,
                               RemoveOptionsAndTerminatePhase>::value,
                  "The last action in the initialization phase must be "
                  "Initialization::Actions::RemoveOptionsAndTerminatePhase.");
    return std::make_tuple(
        db::create_from<tags_to_remove_this_time>(std::move(box)), true);
  }
};
}  // namespace Actions
}  // namespace Initialization
