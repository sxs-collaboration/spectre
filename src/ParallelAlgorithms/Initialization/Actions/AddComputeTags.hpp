// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
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
 *
 * \brief Initialize the list of compute tags in `ComputeTagsList`
 *
 * Uses: nothing
 *
 * DataBox changes:
 * - Adds:
 *   - Each compute tag in the list `ComputeTagsList`
 * - Removes:
 *   - nothing
 * - Modifies:
 *   - nothing
 */
template <typename ComputeTagsList>
struct AddComputeTags {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::make_tuple(
        merge_into_databox<AddComputeTags, db::AddSimpleTags<>,
                           db::AddComputeTags<ComputeTagsList>>(
            std::move(box)));
  }
};
}  // namespace Actions
}  // namespace Initialization
