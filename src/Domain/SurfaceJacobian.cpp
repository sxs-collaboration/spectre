// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/SurfaceJacobian.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace domain {

void surface_jacobian(
    const gsl::not_null<Scalar<DataVector>*> surface_jacobian,
    const Scalar<DataVector>& det_jacobian_on_face,
    const Scalar<DataVector>& face_normal_magnitude) noexcept {
  get(*surface_jacobian) =
      get(det_jacobian_on_face) * get(face_normal_magnitude);
}

Scalar<DataVector> surface_jacobian(
    const Scalar<DataVector>& det_jacobian_on_face,
    const Scalar<DataVector>& face_normal_magnitude) noexcept {
  Scalar<DataVector> surface_jacobian{get(det_jacobian_on_face).size()};
  domain::surface_jacobian(make_not_null(&surface_jacobian),
                           det_jacobian_on_face, face_normal_magnitude);
  return surface_jacobian;
}

template <size_t Dim, typename TargetFrame>
void surface_jacobian(
    const gsl::not_null<Scalar<DataVector>*> surface_jacobian,
    const ElementMap<Dim, TargetFrame>& element_map,
    const Mesh<Dim - 1>& face_mesh, const Direction<Dim>& direction,
    const Scalar<DataVector>& face_normal_magnitude) noexcept {
  domain::surface_jacobian(
      surface_jacobian,
      determinant(element_map.jacobian(
          interface_logical_coordinates(face_mesh, direction))),
      face_normal_magnitude);
}

template <size_t Dim, typename TargetFrame>
Scalar<DataVector> surface_jacobian(
    const ElementMap<Dim, TargetFrame>& element_map,
    const Mesh<Dim - 1>& face_mesh, const Direction<Dim>& direction,
    const Scalar<DataVector>& face_normal_magnitude) noexcept {
  Scalar<DataVector> surface_jacobian{face_mesh.number_of_grid_points()};
  domain::surface_jacobian(make_not_null(&surface_jacobian), element_map,
                           face_mesh, direction, face_normal_magnitude);
  return surface_jacobian;
}

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(r, data)                                     \
  template void surface_jacobian(                                \
      gsl::not_null<Scalar<DataVector>*> surface_jacobian,       \
      const ElementMap<DIM(data), FRAME(data)>& element_map,     \
      const Mesh<DIM(data) - 1>& face_mesh,                      \
      const Direction<DIM(data)>& direction,                     \
      const Scalar<DataVector>& face_normal_magnitude) noexcept; \
  template Scalar<DataVector> surface_jacobian(                  \
      const ElementMap<DIM(data), FRAME(data)>& element_map,     \
      const Mesh<DIM(data) - 1>& face_mesh,                      \
      const Direction<DIM(data)>& direction,                     \
      const Scalar<DataVector>& face_normal_magnitude) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Inertial))

#undef DIM
#undef INSTANTIATE
/// \endcond

}  // namespace domain
