// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Utilities/GetOutput.hpp"

/// \cond
namespace domain {
template <typename SourceFrame, typename TargetFrame, size_t Dim>
class CoordinateMapBase;
}  // namespace domain
/// \endcond

namespace domain {
namespace CoordinateMaps {
/// \ingroup ComputationalDomainGroup
/// \brief %Tags for the coordinate maps.
namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// The coordinate map from source to target coordinates
template <size_t VolumeDim, typename SourceFrame, typename TargetFrame>
struct CoordinateMap : db::SimpleTag {
static constexpr size_t dim = VolumeDim;
  using target_frame = TargetFrame;
  using source_frame = SourceFrame;

  static std::string name() noexcept {
    return "CoordinateMap(" + get_output(SourceFrame{}) + "," +
           get_output(TargetFrame{}) + ")";
  }
  using type = std::unique_ptr<
      domain::CoordinateMapBase<SourceFrame, TargetFrame, VolumeDim>>;
};
}  // namespace Tags
}  // namespace CoordinateMaps
}  // namespace domain
