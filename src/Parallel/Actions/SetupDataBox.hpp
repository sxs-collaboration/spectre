// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Actions {
/// \cond
struct SetupDataBox;
/// \endcond

namespace detail {
template <typename Action, typename enable = std::void_t<>>
struct optional_simple_tags {
  using type = tmpl::list<>;
};

template <typename Action>
struct optional_simple_tags<Action, std::void_t<typename Action::simple_tags>> {
  using type = typename Action::simple_tags;
};

template <typename Action, typename enable = std::void_t<>>
struct optional_compute_tags {
  using type = tmpl::list<>;
};

template <typename Action>
struct optional_compute_tags<Action,
                             std::void_t<typename Action::compute_tags>> {
  using type = typename Action::compute_tags;
};

template <typename ActionList>
using get_action_list_simple_tags = tmpl::remove_duplicates<
    tmpl::flatten<tmpl::transform<ActionList, optional_simple_tags<tmpl::_1>>>>;

template <typename ActionList>
using get_action_list_compute_tags = tmpl::remove_duplicates<tmpl::flatten<
    tmpl::transform<ActionList, optional_compute_tags<tmpl::_1>>>>;

template <typename DbTags, typename... SimpleTags, typename... ComputeTags>
auto merge_into_databox_helper(db::DataBox<DbTags>&& box,
                               tmpl::list<SimpleTags...> /*meta*/,
                               tmpl::list<ComputeTags...> /*meta*/) noexcept {
  return db::create_from<db::RemoveTags<>, db::AddSimpleTags<SimpleTags...>,
                         db::AddComputeTags<ComputeTags...>>(
      std::move(box), typename SimpleTags::type{}...);
}
}  // namespace detail

/*!
 * \brief Add into the \ref DataBoxGroup "DataBox" default constructed items for
 * the collection of tags requested by any of the actions in the current phase.
 *
 * \details This action adds all of the simple tags given in the `simple_tags`
 * type lists in each of the other actions in the current phase, and all of the
 * compute tags given in the `compute_tags` type lists. If an action does not
 * give either of the type lists, it is treated as an empty type list.
 *
 * To prevent the proliferation of many \ref DataBoxGroup "DataBox" types, which
 * can drastically slow compile times, it is preferable to use only this action
 * to add tags to the \ref DataBoxGroup "DataBox", and place this action at the
 * start of the `Initialization` phase action list. The rest of the
 * initialization actions should specify `simple_tags` and `compute_tags`, and
 * assign initial values to those tags, but not add those tags into the \ref
 * DataBoxGroup "DataBox".
 *
 * An example initialization action:
 * \snippet Test_SetupDataBox.cpp initialization_action
 */
struct SetupDataBox {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using action_list_simple_tags =
        typename detail::get_action_list_simple_tags<ActionList>;
    using action_list_compute_tags =
        typename detail::get_action_list_compute_tags<ActionList>;
    // grab the simple_tags, compute_tags, mutate the databox, creating
    // default-constructed objects.
    return std::make_tuple(detail::merge_into_databox_helper(
        std::move(box),
        tmpl::list_difference<action_list_simple_tags,
                              typename db::DataBox<DbTags>::simple_item_tags>{},
        tmpl::list_difference<
            action_list_compute_tags,
            typename db::DataBox<DbTags>::compute_item_tags>{}));
  }
};
}  // namespace Actions
