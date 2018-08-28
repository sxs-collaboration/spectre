// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>

#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db
/// \endcond

namespace Actions {
/// \ingroup ActionsGroup
/// Labels a location in the action list that can be jumped to using Goto.
///
/// Uses:
/// - ConstGlobalCache: nothing
/// - DataBox: nothing
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies: nothing
template <typename Tag>
struct Label {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    static_assert(
        tmpl::count_if<ActionList,
                       std::is_same<tmpl::_1, tmpl::pin<Label<Tag>>>>::value ==
            1,
        "Duplicate label");
    return std::forward_as_tuple(std::move(box));
  }
};

/// \ingroup ActionsGroup
/// Jumps to a Label.
///
/// Uses:
/// - ConstGlobalCache: nothing
/// - DataBox: nothing
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies: nothing
template <typename Tag>
struct Goto {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    constexpr size_t index =
        tmpl::index_of<ActionList, Actions::Label<Tag>>::value;
    return std::tuple<db::DataBox<DbTags>&&, bool, size_t>(std::move(box),
                                                           false, index);
  }
};
}  // namespace Actions
