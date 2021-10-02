// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
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
/// - GlobalCache: nothing
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
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
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
/// - GlobalCache: nothing
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
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    constexpr size_t index =
        tmpl::index_of<ActionList, Actions::Label<Tag>>::value;
    return std::tuple<db::DataBox<DbTags>&&, bool, size_t>(std::move(box),
                                                           false, index);
  }
};

namespace Goto_detail {

template <typename ConditionTag>
struct RepeatEnd;

template <typename ConditionTag>
struct RepeatStart {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool, size_t> apply(
      db::DataBox<DbTagsList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    return {std::move(box), false,
            db::get<ConditionTag>(box)
                ? tmpl::index_of<ActionList, RepeatEnd<ConditionTag>>::value + 1
                : tmpl::index_of<ActionList, RepeatStart>::value + 1};
  }
};

template <typename ConditionTag>
struct RepeatEnd {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool, size_t> apply(
      db::DataBox<DbTagsList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    return {
        std::move(box), false,
        db::get<ConditionTag>(box)
            ? tmpl::index_of<ActionList, RepeatEnd>::value + 1
            : tmpl::index_of<ActionList, RepeatStart<ConditionTag>>::value + 1};
  }
};

}  // namespace Goto_detail

/// Repeats the `ActionList` until `ConditionTag` is `True`.
template <typename ConditionTag, typename ActionList>
using RepeatUntil =
    tmpl::flatten<tmpl::list<Goto_detail::RepeatStart<ConditionTag>, ActionList,
                             Goto_detail::RepeatEnd<ConditionTag>>>;
}  // namespace Actions
