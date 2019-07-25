// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "ErrorHandling/Error.hpp"
#include "ParallelBackend/Actions/TerminatePhase.hpp"
#include "ParallelBackend/AddOptionsToDataBox.hpp"
#include "ParallelBackend/ConstGlobalCache.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Initialization {

namespace detail {
template <typename Action, typename = cpp17::void_t<>>
struct get_options_tags {
  using type = tmpl::list<>;
};

template <typename Action>
struct get_options_tags<
    Action, cpp17::void_t<typename Action::initialization_option_tags>> {
  using type = typename Action::initialization_option_tags;
};
}  // namespace detail

/// \ingroup InitializationGroup
/// Get the list of input file options from the list of initialization actions.
template <typename InitializationActionList>
using option_tags = tmpl::remove_duplicates<tmpl::flatten<tmpl::transform<
    InitializationActionList, detail::get_options_tags<tmpl::_1>>>>;

/// \ingroup InitializationGroup
/// Available actions to be used in the initialization.
///
/// The action list for initialization must end with the
/// `Initialization::Actions::RemoveOptionsAndTerminatePhase` action. For
/// example,
///
/// \snippet Test_RemoveOptionsAndTerminatePhase.cpp actions
///
/// The parallel component's `add_options_to_databox` will typically just be:
///
/// \snippet Test_RemoveOptionsAndTerminatePhase.cpp options_to_databox
///
namespace Actions {
/// \ingroup InitializationGroup
struct RemoveOptionsAndTerminatePhase {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ArrayIndex,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using options_tags_to_remove = option_tags<ActionList>;
    // Retrieve the `initialization_option_tags` that are still in the DataBox
    // and remove them.
    using options_tags_to_remove_this_time = tmpl::filter<
        options_tags_to_remove,
        tmpl::bind<
            tmpl::list_contains,
            tmpl::pin<typename db::DataBox<DbTagsList>::simple_item_tags>,
            tmpl::_1>>;
    static_assert(std::is_same<tmpl::back<ActionList>,
                               RemoveOptionsAndTerminatePhase>::value,
                  "The last action in the initialization phase must be "
                  "Initialization::Actions::RemoveOptionsAndTerminatePhase.");
    return std::make_tuple(
        db::create_from<options_tags_to_remove_this_time>(std::move(box)),
        true);
  }
};
}  // namespace Actions
}  // namespace Initialization
