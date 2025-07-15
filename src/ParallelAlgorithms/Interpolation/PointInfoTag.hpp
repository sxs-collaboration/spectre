// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TMPL.hpp"

namespace intrp::Tags {
/*!
 * \brief  PointInfo holds the points to be interpolated onto,
 * in whatever frame those points are to be held constant.
 *
 * \details PointInfo is used only for interpolation points that are
 * time-independent in some frame, so that there is no `Interpolator`
 * ParallelComponent. `VolumeDim` is a typename rather than a `size_t` to better
 * facilitate this tag being used in metafunctions. `VolumeDim` must be a
 * `tmpl::size_t`.
 */
template <typename InterpolationTargetTag, typename VolumeDim>
struct PointInfo : db::SimpleTag {
  using type =
      tnsr::I<DataVector, VolumeDim::value,
              typename InterpolationTargetTag::compute_target_points::frame>;
};
}  // namespace intrp::Tags
