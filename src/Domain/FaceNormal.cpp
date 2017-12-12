// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/FaceNormal.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Direction.hpp"
#include "Domain/LogicalCoordinates.hpp"

template <size_t VolumeDim>
tnsr::i<DataVector, VolumeDim, Frame::Grid> unnormalized_face_normal(
    const Index<VolumeDim - 1>& interface_extents,
    const CoordinateMapBase<Frame::Logical, Frame::Grid, VolumeDim>& map,
    const Direction<VolumeDim>& direction) noexcept {
  const auto interface_coords =
      interface_logical_coordinates(interface_extents, direction);
  const auto inv_jacobian_on_interface = map.inv_jacobian(interface_coords);

  const auto sliced_away_dim = direction.dimension();
  const double sign = direction.sign();

  tnsr::i<DataVector, VolumeDim, Frame::Grid> face_normal(
      interface_extents.product());

  for (size_t d = 0; d < VolumeDim; ++d) {
    face_normal.get(d) =
        sign * inv_jacobian_on_interface.get(sliced_away_dim, d);
  }
  return face_normal;
}

template tnsr::i<DataVector, 1, Frame::Grid> unnormalized_face_normal(
    const Index<0>& interface_extents,
    const CoordinateMapBase<Frame::Logical, Frame::Grid, 1>& map,
    const Direction<1>& direction) noexcept;

template tnsr::i<DataVector, 2, Frame::Grid> unnormalized_face_normal(
    const Index<1>& interface_extents,
    const CoordinateMapBase<Frame::Logical, Frame::Grid, 2>& map,
    const Direction<2>& direction) noexcept;

template tnsr::i<DataVector, 3, Frame::Grid> unnormalized_face_normal(
    const Index<2>& interface_extents,
    const CoordinateMapBase<Frame::Logical, Frame::Grid, 3>& map,
    const Direction<3>& direction) noexcept;
