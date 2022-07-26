// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Initialization {
namespace detail {
template <typename... TagsToRemove, typename BoxTags>
constexpr void remove(const gsl::not_null<db::DataBox<BoxTags>*> box,
                      tmpl::list<TagsToRemove...> /*meta*/) {
  EXPAND_PACK_LEFT_TO_RIGHT(db::remove<TagsToRemove>(box));
}
}  // namespace detail

/// \ingroup InitializationGroup
/// Removes an item from the DataBox by resetting its value to that given
/// by default initialization.  In debug builds, using a removed item throws an
/// exception.
///
/// \snippet Test_RemoveOptionsAndTerminatePhase.cpp actions
///
namespace Actions {
/// \ingroup InitializationGroup
struct RemoveOptionsAndTerminatePhase {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ArrayIndex,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    using initialization_tags = Parallel::get_initialization_tags<ActionList>;
    using initialization_tags_to_keep =
        Parallel::get_initialization_tags_to_keep<ActionList>;
    // Keep the tags that are in initialization_tags_to_keep.
    using tags_to_remove = tmpl::remove_if<
        initialization_tags,
        tmpl::bind<tmpl::list_contains, tmpl::pin<initialization_tags_to_keep>,
                   tmpl::_1>>;
    if constexpr (tmpl::size<tags_to_remove>::value > 0) {
      detail::remove(make_not_null(&box), tags_to_remove{});
    }
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};
}  // namespace Actions
}  // namespace Initialization
