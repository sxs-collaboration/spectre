// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/FaceNormal.hpp"

#include "DataStructures/DataVector.hpp"            // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"         // IWYU pragma: keep
#include "Domain/CoordinateMaps/CoordinateMap.hpp"  // IWYU pragma: keep
#include "Domain/Direction.hpp"                     // IWYU pragma: keep
#include "Domain/ElementMap.hpp"                    // IWYU pragma: keep
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"                          // IWYU pragma: keep
#include "Utilities/GenerateInstantiations.hpp"

namespace {
template <typename TargetFrame, size_t VolumeDim, typename Map>
tnsr::i<DataVector, VolumeDim, TargetFrame> unnormalized_face_normal_impl(
    const Mesh<VolumeDim - 1>& interface_mesh, const Map& map,
    const Direction<VolumeDim>& direction) noexcept {
  auto interface_coords =
      interface_logical_coordinates(interface_mesh, direction);
  const auto inv_jacobian_on_interface =
      map.inv_jacobian(std::move(interface_coords));

  const auto sliced_away_dim = direction.dimension();
  const double sign = direction.sign();

  tnsr::i<DataVector, VolumeDim, TargetFrame> face_normal(
      interface_mesh.number_of_grid_points());

  for (size_t d = 0; d < VolumeDim; ++d) {
    face_normal.get(d) =
        sign * inv_jacobian_on_interface.get(sliced_away_dim, d);
  }
  return face_normal;
}
}  // namespace

template <size_t VolumeDim, typename TargetFrame>
tnsr::i<DataVector, VolumeDim, TargetFrame> unnormalized_face_normal(
    const Mesh<VolumeDim - 1>& interface_mesh,
    const ElementMap<VolumeDim, TargetFrame>& map,
    const Direction<VolumeDim>& direction) noexcept {
  return unnormalized_face_normal_impl<TargetFrame>(interface_mesh, map,
                                                    direction);
}

template <size_t VolumeDim, typename TargetFrame>
tnsr::i<DataVector, VolumeDim, TargetFrame> unnormalized_face_normal(
    const Mesh<VolumeDim - 1>& interface_mesh,
    const domain::CoordinateMapBase<Frame::Logical, TargetFrame, VolumeDim>&
        map,
    const Direction<VolumeDim>& direction) noexcept {
  return unnormalized_face_normal_impl<TargetFrame>(interface_mesh, map,
                                                    direction);
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define GET_FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(_, data)                                                \
  template tnsr::i<DataVector, GET_DIM(data), GET_FRAME(data)>                \
  unnormalized_face_normal(const Mesh<GET_DIM(data) - 1>&,                    \
                           const ElementMap<GET_DIM(data), GET_FRAME(data)>&, \
                           const Direction<GET_DIM(data)>&) noexcept;         \
  template tnsr::i<DataVector, GET_DIM(data), GET_FRAME(data)>                \
  unnormalized_face_normal(                                                   \
      const Mesh<GET_DIM(data) - 1>&,                                         \
      const domain::CoordinateMapBase<Frame::Logical, GET_FRAME(data),        \
                                      GET_DIM(data)>&,                        \
      const Direction<GET_DIM(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3),
                        (Frame::Inertial, Frame::Grid))

#undef GET_DIM
#undef GET_FRAME
#undef INSTANTIATION
