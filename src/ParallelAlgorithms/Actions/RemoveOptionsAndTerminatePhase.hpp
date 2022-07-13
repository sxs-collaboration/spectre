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
namespace detail {
template <typename... TagsToDefaultAssign, typename BoxTags>
constexpr void default_assign(const gsl::not_null<db::DataBox<BoxTags>*> box,
                              tmpl::list<TagsToDefaultAssign...> /*meta*/) {
  db::mutate<TagsToDefaultAssign...>(box, [](const auto... args) {
    EXPAND_PACK_LEFT_TO_RIGHT(*args = typename TagsToDefaultAssign::type{});
  });
}
}  // namespace detail

/// \ingroup InitializationGroup
/// Removes an option from the DataBox by resetting its value to that given
/// by default initialization.
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
      detail::default_assign(make_not_null(&box), tags_to_remove{});
    }
    return std::make_tuple(std::move(box), true);
  }
};
}  // namespace Actions
}  // namespace Initialization
