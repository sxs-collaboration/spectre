// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/optional.hpp>
#include <vector>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/IdPair.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Structure/BlockId.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace intrp {
namespace Vars {
template <typename InterpolationTargetTag, size_t VolumeDim>
struct PointInfoTag {
  /// This is the type returned from BlockLogicalCoords.
  /// It encodes the list of all points (in block
  /// logical coordinates) that need to be interpolated onto for a
  /// given `InterpolationTarget`.  Only a subset of those points
  /// will be contained in the `Element` that uses this Tag.
  using type = std::vector<boost::optional<IdPair<
      domain::BlockId, tnsr::I<double, VolumeDim, typename ::Frame::Logical>>>>;
};
}  // namespace Vars

namespace Tags {

namespace point_info_detail {
template <typename InterpolationTargetTag, typename Metavariables>
using WrappedPointInfoTag =
    Vars::PointInfoTag<InterpolationTargetTag, Metavariables::volume_dim>;
}  // namespace point_info_detail

/// The following tag is for the case in which interpolation
/// bypasses the `Interpolator` ParallelComponent.  In that case,
/// the `Element` must hold interpolation information in its `DataBox`.
///
/// A particular `Vars::PointInfo` can be retrieved from this
/// `TaggedTuple` via a `Vars::PointInfoTag`.
template <typename Metavariables>
struct InterpPointInfo : db::SimpleTag {
  using type = tuples::tagged_tuple_from_typelist<db::wrap_tags_in<
      point_info_detail::WrappedPointInfoTag,
      typename Metavariables::interpolation_target_tags, Metavariables>>;
};

}  // namespace Tags
}  // namespace intrp
