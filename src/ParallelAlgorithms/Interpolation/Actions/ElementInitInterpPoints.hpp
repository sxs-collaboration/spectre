// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
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

namespace intrp::Actions {
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
///   - `intrp::Tags::PointInfo` for each non-sequential target tag
/// - Removes: nothing
/// - Modifies: nothing
template <size_t VolumeDim, typename AllInterpolationTargetTags>
struct ElementInitInterpPoints {
  using simple_tags = tmpl::transform<
      intrp::InterpolationTarget_detail::get_non_sequential_target_tags<
          AllInterpolationTargetTags>,
      tmpl::bind<intrp::Tags::PointInfo, tmpl::_1,
                 tmpl::pin<tmpl::size_t<VolumeDim>>>>;
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& /*box*/,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    // Here we only want the `intrp::Tags::PointInfo` default constructed
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace intrp::Actions
