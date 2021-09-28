// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "ParallelAlgorithms/Interpolation/PointInfoTag.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
/// \endcond

namespace intrp {
namespace Actions {

/// \ingroup ActionsGroup
/// \brief Adds interpolation point holders to the Element's DataBox.
///
/// This action is for the case in which the points are time-independent.
///
/// This action should be placed in the Initialization PDAL for DgElementArray.
///
/// Uses: nothing
///
/// DataBox changes:
/// - Adds:
///   - `intrp::Tags::InterpPointInfo<Metavariables>`
/// - Removes: nothing
/// - Modifies: nothing
///
/// \note This action relies on the `SetupDataBox` aggregated initialization
/// mechanism, so `Actions::SetupDataBox` must be present in the
/// `Initialization` phase action list prior to this action.
template <typename InterpPointInfoTag>
struct ElementInitInterpPoints {
  using simple_tags = tmpl::list<InterpPointInfoTag>;
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    // Here we only want the `InterpPointInfoTag` default constructed, which was
    // done in `SetupDataBox`
    return std::make_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace intrp
