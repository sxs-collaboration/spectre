// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/FaceNormal.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Direction.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace {
template <typename TargetFrame, size_t VolumeDim, typename Map>
tnsr::i<DataVector, VolumeDim, TargetFrame> unnormalized_face_normal_impl(
    const Index<VolumeDim - 1>& interface_extents, const Map& map,
    const Direction<VolumeDim>& direction) noexcept {
  auto interface_coords =
      interface_logical_coordinates(interface_extents, direction);
  const auto inv_jacobian_on_interface =
      map.inv_jacobian(std::move(interface_coords));

  const auto sliced_away_dim = direction.dimension();
  const double sign = direction.sign();

  tnsr::i<DataVector, VolumeDim, TargetFrame> face_normal(
      interface_extents.product());

  for (size_t d = 0; d < VolumeDim; ++d) {
    face_normal.get(d) =
        sign * inv_jacobian_on_interface.get(sliced_away_dim, d);
  }
  return face_normal;
}
}  // namespace

template <size_t VolumeDim, typename TargetFrame>
tnsr::i<DataVector, VolumeDim, TargetFrame> unnormalized_face_normal(
    const Index<VolumeDim - 1>& interface_extents,
    const ElementMap<VolumeDim, TargetFrame>& map,
    const Direction<VolumeDim>& direction) noexcept {
  return unnormalized_face_normal_impl<TargetFrame>(interface_extents, map,
                                                    direction);
}

template <size_t VolumeDim, typename TargetFrame>
tnsr::i<DataVector, VolumeDim, TargetFrame> unnormalized_face_normal(
    const Index<VolumeDim - 1>& interface_extents,
    const CoordinateMapBase<Frame::Logical, TargetFrame, VolumeDim>& map,
    const Direction<VolumeDim>& direction) noexcept {
  return unnormalized_face_normal_impl<TargetFrame>(interface_extents, map,
                                                    direction);
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define GET_FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(_, data)                                                 \
  template tnsr::i<DataVector, GET_DIM(data), GET_FRAME(data)>                 \
  unnormalized_face_normal(                                                    \
      const Index<GET_DIM(data) - 1>& interface_extents,                       \
      const ElementMap<GET_DIM(data), GET_FRAME(data)>& map,                   \
      const Direction<GET_DIM(data)>& direction) noexcept;                     \
  template tnsr::i<DataVector, GET_DIM(data), GET_FRAME(data)>                 \
  unnormalized_face_normal(                                                    \
      const Index<GET_DIM(data) - 1>& interface_extents,                       \
      const CoordinateMapBase<Frame::Logical, GET_FRAME(data), GET_DIM(data)>& \
          map,                                                                 \
      const Direction<GET_DIM(data)>& direction) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3),
                        (Frame::Inertial, Frame::Grid))

#undef GET_DIM
#undef GET_FRAME
#undef INSTANTIATION
