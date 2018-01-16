// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Utilities/Requires.hpp"

namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples

namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel

namespace Actions {
/// \ingroup ActionsGroup
/// \brief Compute the volume time derivative of the variables
///
/// \note The DataBox items holding the time derivatives are not zeroed
///
/// Uses:
/// - DataBox:
///   - All elements in `system::du_dt::return_tags`
///   - All elements in `system::du_dt::argument_tags`
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies: db::add_tag_prefix<Tags::dt, typename system::variables_tag>
template <size_t Dim>
struct ComputeVolumeDuDt {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl::size<DbTagsList>::value != 0> = nullptr>
  static auto apply(db::DataBox<DbTagsList>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;
    // Note: dt_variables is not zeroed and du_dt cannot assume this.
    db::mutate_apply<typename system::du_dt::return_tags,
                     typename system::du_dt::argument_tags>(
        typename system::du_dt{}, box);
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
