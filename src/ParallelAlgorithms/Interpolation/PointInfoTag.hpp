// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <vector>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/IdPair.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Structure/BlockId.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace intrp {
namespace Vars {
/// PointInfoTag holds the points to be interpolated onto,
/// in whatever frame those points are to be held constant.
/// PointInfoTag is used only for interpolation points that are
/// time-independent in some frame, so that there is no `Interpolator`
/// ParallelComponent.
template <typename InterpolationTargetTag, size_t VolumeDim>
struct PointInfoTag {
  using type =
      tnsr::I<DataVector, VolumeDim,
              typename InterpolationTargetTag::compute_target_points::frame>;
};
}  // namespace Vars

namespace Tags {

namespace point_info_detail {
template <typename InterpolationTargetTag, typename Metavariables>
using WrappedPointInfoTag =
    Vars::PointInfoTag<InterpolationTargetTag, Metavariables::volume_dim>;
}  // namespace point_info_detail

/// Base tag for `InterpPointInfo` so we don't have to have the metavariables in
/// order to retrieve the tag
struct InterpPointInfoBase : db::BaseTag {};

/// The following tag is for the case in which interpolation
/// bypasses the `Interpolator` ParallelComponent.  In that case,
/// the `Element` must hold interpolation information in its `DataBox`.
///
/// A particular `Vars::PointInfo` can be retrieved from this
/// `TaggedTuple` via a `Vars::PointInfoTag`.
template <typename Metavariables>
struct InterpPointInfo : db::SimpleTag, InterpPointInfoBase {
  using type = tuples::tagged_tuple_from_typelist<db::wrap_tags_in<
      point_info_detail::WrappedPointInfoTag,
      typename Metavariables::interpolation_target_tags, Metavariables>>;
};

}  // namespace Tags
}  // namespace intrp
