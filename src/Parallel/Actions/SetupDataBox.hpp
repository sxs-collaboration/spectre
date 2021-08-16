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
struct get_action_list_simple_tags {
  using type = tmpl::flatten<tmpl::transform<typename ActionList::action_list,
                                             optional_simple_tags<tmpl::_1>>>;
};

template <typename ActionList>
struct get_action_list_compute_tags {
  using type = tmpl::flatten<tmpl::transform<typename ActionList::action_list,
                                             optional_compute_tags<tmpl::_1>>>;
};

template <typename Pdal>
using get_pdal_simple_tags =
    tmpl::flatten<tmpl::transform<Pdal, get_action_list_simple_tags<tmpl::_1>>>;

template <typename Pdal>
using get_pdal_compute_tags = tmpl::flatten<
    tmpl::transform<Pdal, get_action_list_compute_tags<tmpl::_1>>>;

template <typename DbTags, typename... SimpleTags, typename... ComputeTags>
auto merge_into_databox_helper(db::DataBox<DbTags>&& box,
                               tmpl::list<SimpleTags...> /*meta*/,
                               tmpl::list<ComputeTags...> /*meta*/) noexcept {
  return db::create_from<db::RemoveTags<>, db::AddSimpleTags<SimpleTags...>,
                         db::AddComputeTags<ComputeTags...>>(std::move(box));
}
}  // namespace detail

/*!
 * \brief Add into the \ref DataBoxGroup "DataBox" default constructed items for
 * the collection of tags requested by any of the actions in the phase-dependent
 * action list.
 *
 * \details This action adds all of the simple tags given in the `simple_tags`
 * type lists in each of the other actions in the current component's full
 * phase-dependent action list, and all of the compute tags given in the
 * `compute_tags` type lists. If an action does not give either of the type
 * lists, it is treated as an empty type list.
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
 *
 * \note This action operates on the assumption that the phase dependent action
 * list of the `ParallelComponent` and the `ActionList` do not depend on the
 * \ref DataBoxGroup "DataBox" type in the Algorithm. This assumption holds
 * for all current utilities, but if it must be relaxed, revisions to
 * `SetupDataBox` may be required to avoid a cyclic dependency of the \ref
 * DataBoxGroup "DataBox" types.
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
        tmpl::remove_duplicates<detail::get_pdal_simple_tags<
            typename ParallelComponent::phase_dependent_action_list>>;
    using action_list_compute_tags =
        tmpl::remove_duplicates<detail::get_pdal_compute_tags<
            typename ParallelComponent::phase_dependent_action_list>>;
    using all_new_tags =
        tmpl::append<action_list_simple_tags, action_list_compute_tags>;
    // We just check the first tag in the list to prevent repeat-applications
    // of the action. Any cases where a tag is mistakenly added to the DataBox
    // before `SetupDataBox` is called (which shouldn't happen if it's used as
    // suggested) will result in a multiple inheritance error.
    if constexpr (tmpl::size<all_new_tags>::value != 0_st) {
      if constexpr (not tmpl::list_contains_v<DbTags,
                                              tmpl::front<all_new_tags>>) {
        return std::make_tuple(detail::merge_into_databox_helper(
            std::move(box), action_list_simple_tags{},
            action_list_compute_tags{}));
      } else {
        ERROR(
            "Trying to call SetupDataBox after it has already been called "
            "previously. SetupDataBox may only be called once.");
        return std::make_tuple(std::move(box));
      }
    } else {
      return std::make_tuple(std::move(box));
    }

  }
};
}  // namespace Actions
