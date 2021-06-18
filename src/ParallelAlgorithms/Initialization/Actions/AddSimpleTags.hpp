// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
/// \endcond

namespace Initialization {
namespace Actions {
/*!
 * \ingroup ActionsGroup
 * \brief Initialize the list of simple tags in `Mutators::return_tags` by
 * calling each mutator in the order they are specified
 *
 * There's a specialization for `AddSimpleTags<tmpl::list<Mutators...>>` that
 * can also be used if a `tmpl::list` is available.
 *
 * Uses: nothing
 *
 * DataBox changes:
 * - Adds:
 *   - Each simple tag in the list `Mutators::return_tags`
 * - Removes:
 *   - nothing
 * - Modifies:
 *   - Each simple tag in the list `Mutators::return_tags`
 *
 * \note This action relies on the `SetupDataBox` aggregated initialization
 * mechanism, so `Actions::SetupDataBox` must be present in the `Initialization`
 * phase action list prior to this action.
 */
template <typename... Mutators>
struct AddSimpleTags {
  using simple_tags =
      tmpl::flatten<tmpl::append<typename Mutators::return_tags...>>;
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    EXPAND_PACK_LEFT_TO_RIGHT(db::mutate_apply<Mutators>(make_not_null(&box)));
    return std::make_tuple(std::move(box));
  }
};

/// \cond
template <typename... Mutators>
struct AddSimpleTags<tmpl::list<Mutators...>>
    : public AddSimpleTags<Mutators...> {};
/// \endcond
}  // namespace Actions
}  // namespace Initialization
