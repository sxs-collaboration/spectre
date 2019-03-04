// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples

namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
/// \endcond

namespace db {
namespace Actions {
/*!
 * \ingroup ActionsGroup
 * \brief Apply the function `F::apply` to the DataBox
 *
 * The function `F::apply` is invoked with the `F::argument_tags`. The result
 * of this computation is stored in the `F::return_tags`.
 *
 * Uses:
 * - DataBox:
 *   - All elements in `F::argument_tags`
 *   - All elements in `F::return_tags`
 *
 * DataBox changes:
 * - Modifies:
 *   - All elements in `F::return_tags`
 */
template <typename F>
struct MutateApply {
  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            Requires<tmpl::size<DbTagsList>::value != 0> = nullptr>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    db::mutate_apply<F>(make_not_null(&box));
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace db
