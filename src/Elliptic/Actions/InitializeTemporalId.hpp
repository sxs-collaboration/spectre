// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"

/// \cond
template <size_t VolumeDim>
class ElementIndex;
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace elliptic {
namespace Actions {

/*!
 * \brief Initializes the DataBox tag for the temporal ID.
 *
 * Currently, this action simply constructs the temporal ID from a zero integer.
 * This is suitable for elliptic iteration IDs.
 *
 * Uses:
 * - Metavariables:
 *   - `temporal_id`
 *
 * DataBox:
 * - Adds:
 *   - `temporal_id`
 *   - `Tags::Next<temporal_id>`
 */
struct InitializeTemporalId {
  template <typename DataBox, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(DataBox& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using temporal_id_tag = typename Metavariables::temporal_id;
    db::item_type<temporal_id_tag> temporal_id{0};
    return std::make_tuple(
        ::Initialization::merge_into_databox<
            InitializeTemporalId, db::AddSimpleTags<temporal_id_tag>,
            db::AddComputeTags<::Tags::NextCompute<temporal_id_tag>>>(
            std::move(box), std::move(temporal_id)));
  }
};

}  // namespace Actions
}  // namespace elliptic
