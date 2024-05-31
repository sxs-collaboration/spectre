// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/DiscontinuousGalerkin/Initialization.hpp"

#include <array>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/IndexToSliceAt.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/ProjectToBoundary.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace elliptic::dg {

namespace detail{
template <size_t Dim>
void initialize_coords_and_jacobians(
    gsl::not_null<tnsr::I<DataVector, Dim, Frame::ElementLogical>*>
        logical_coords,
    gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> inertial_coords,
    gsl::not_null<InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                                  Frame::Inertial>*>
        inv_jacobian,
    gsl::not_null<Scalar<DataVector>*> det_inv_jacobian,
    gsl::not_null<Scalar<DataVector>*> det_jacobian,
    gsl::not_null<InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                                  Frame::Inertial>*>
        det_times_inv_jacobian,
        const Mesh<Dim>& mesh,
    const ElementMap<Dim, Frame::Inertial>& element_map,
    const domain::FunctionsOfTimeMap& functions_of_time) {
  // Coordinates
  *logical_coords = logical_coordinates(mesh);
  *inertial_coords =
      element_map(*logical_coords, 0., functions_of_time);
  // Jacobian
  // Note: we can try to use `::dg::metric_identity_jacobian_quantities` here.
  // When I tried (NV, Dec 2023) the DG residuals diverged on a sphere domain
  // with a large outer boundary (1e9). This was possibly because no
  // metric-identity Jacobians were used on faces, though I also tried slicing
  // the metric-identity Jacobian to the faces and that didn't help.
  *inv_jacobian =
      element_map.inv_jacobian(*logical_coords, 0., functions_of_time);
  *det_inv_jacobian = determinant(*inv_jacobian);
  get(*det_jacobian) = 1. / get(*det_inv_jacobian);
  *det_times_inv_jacobian = *inv_jacobian;
  for (auto& component : *det_times_inv_jacobian) {
    component *= get(*det_jacobian);
  }
    }
} // namespace detail

template <size_t Dim>
void InitializeGeometry<Dim>::apply(
    const gsl::not_null<Mesh<Dim>*> mesh,
    const gsl::not_null<Element<Dim>*> element,
    const gsl::not_null<DirectionalIdMap<Dim, Mesh<Dim>>*> neighbor_meshes,
    const gsl::not_null<ElementMap<Dim, Frame::Inertial>*> element_map,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::ElementLogical>*>
        logical_coords,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        inertial_coords,
    const gsl::not_null<InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                                        Frame::Inertial>*>
        inv_jacobian,
    const gsl::not_null<Scalar<DataVector>*> det_inv_jacobian,
    const gsl::not_null<Scalar<DataVector>*> det_jacobian,
    const gsl::not_null<InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                                        Frame::Inertial>*>
        det_times_inv_jacobian,
    const std::vector<std::array<size_t, Dim>>& initial_extents,
    const std::vector<std::array<size_t, Dim>>& initial_refinement,
    const Domain<Dim>& domain,
    const domain::FunctionsOfTimeMap& functions_of_time,
    const Spectral::Quadrature quadrature, const ElementId<Dim>& element_id) {
  // Mesh
  ASSERT(quadrature == Spectral::Quadrature::GaussLobatto or
             quadrature == Spectral::Quadrature::Gauss,
         "The elliptic DG scheme supports Gauss and Gauss-Lobatto "
         "grids, but the chosen quadrature is: "
             << quadrature);
  *mesh = domain::Initialization::create_initial_mesh(initial_extents,
                                                      element_id, quadrature);
  // Element
  const auto& block = domain.blocks()[element_id.block_id()];
  *element = domain::Initialization::create_initial_element(element_id, block,
                                                            initial_refinement);
  // Neighbor meshes
  for (const auto& [direction, neighbors] : element->neighbors()) {
    for (const auto& neighbor_id : neighbors) {
      neighbor_meshes->emplace(DirectionalId<Dim>{direction, neighbor_id},
                               domain::Initialization::create_initial_mesh(
                                   initial_extents, neighbor_id, quadrature));
    }
  }
  // Element map
  *element_map = ElementMap<Dim, Frame::Inertial>{element_id, block};
  // Coordinates and Jacobians
  detail::initialize_coords_and_jacobians(
      logical_coords, inertial_coords, inv_jacobian, det_inv_jacobian,
      det_jacobian, det_times_inv_jacobian, *mesh, *element_map,
      functions_of_time);
}

namespace detail {
template <size_t Dim>
void deriv_unnormalized_face_normals_impl(
    const gsl::not_null<DirectionMap<Dim, tnsr::ij<DataVector, Dim>>*>
        deriv_unnormalized_face_normals,
    const Mesh<Dim>& mesh, const Element<Dim>& element,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          Frame::Inertial>& inv_jacobian) {
  if (element.external_boundaries().empty()) {
    return;
  }
  // If the accuracy of this derivative is insufficient we could also compute
  // it on a higher-order grid and then project it down.
  // On Gauss grids we could compute the derivative on a Gauss-Lobatto grid and
  // slice it.
  const auto deriv_inv_jac =
      partial_derivative(inv_jacobian, mesh, inv_jacobian);
  for (const auto& direction : element.external_boundaries()) {
    const auto deriv_inv_jac_on_face =
        ::dg::project_tensor_to_boundary(deriv_inv_jac, mesh, direction);
    auto& deriv_unnormalized_face_normal =
        (*deriv_unnormalized_face_normals)[direction];
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = 0; j < Dim; ++j) {
        deriv_unnormalized_face_normal.get(i, j) =
            direction.sign() *
            deriv_inv_jac_on_face.get(i, direction.dimension(), j);
      }
    }
  }
}

template <size_t Dim>
tnsr::I<DataVector, Dim, Frame::ElementLogical> mortar_logical_coordinates(
    const Mesh<Dim - 1>& mortar_mesh,
    const ::dg::MortarSize<Dim - 1>& mortar_size,
    const Direction<Dim>& direction) {
  auto mortar_logical_coords =
      interface_logical_coordinates(mortar_mesh, direction);
  size_t d_m = 0;
  for (size_t d = 0; d < Dim; ++d) {
    if (d == direction.dimension()) {
      continue;
    }
    if (mortar_size.at(d_m) == Spectral::MortarSize::LowerHalf) {
      mortar_logical_coords.get(d) -= 1.;
      mortar_logical_coords.get(d) *= 0.5;
    } else if (mortar_size.at(d_m) == Spectral::MortarSize::UpperHalf) {
      mortar_logical_coords.get(d) += 1.;
      mortar_logical_coords.get(d) *= 0.5;
    }
    ++d_m;
  }
  return mortar_logical_coords;
}

template <size_t Dim>
void mortar_jacobian(
    const gsl::not_null<Scalar<DataVector>*> mortar_jacobian,
    const gsl::not_null<Scalar<DataVector>*> perpendicular_element_size,
    const Mesh<Dim - 1>& mortar_mesh,
    const ::dg::MortarSize<Dim - 1>& mortar_size,
    const Direction<Dim>& direction,
    const tnsr::I<DataVector, Dim, Frame::ElementLogical>&
        mortar_logical_coords,
    const std::optional<tnsr::II<DataVector, Dim>>& inv_metric_on_mortar,
    const ElementMap<Dim, Frame::Inertial>& element_map,
    const domain::FunctionsOfTimeMap& functions_of_time) {
  determinant(mortar_jacobian, element_map.jacobian(mortar_logical_coords, 0.,
                                                    functions_of_time));
  // These factors of two account for the mortar size
  for (const auto& mortar_size_i : mortar_size) {
    if (mortar_size_i != Spectral::MortarSize::Full) {
      get(*mortar_jacobian) *= 0.5;
    }
  }
  const auto inv_jacobian_on_mortar =
      element_map.inv_jacobian(mortar_logical_coords, 0., functions_of_time);
  const auto unnormalized_mortar_normal =
      unnormalized_face_normal(mortar_mesh, inv_jacobian_on_mortar, direction);
  Scalar<DataVector> mortar_normal_magnitude{};
  if (inv_metric_on_mortar.has_value()) {
    magnitude(make_not_null(&mortar_normal_magnitude),
              unnormalized_mortar_normal, inv_metric_on_mortar.value());
  } else {
    magnitude(make_not_null(&mortar_normal_magnitude),
              unnormalized_mortar_normal);
  }
  get(*mortar_jacobian) *= get(mortar_normal_magnitude);
  get(*perpendicular_element_size) = 2. / get(mortar_normal_magnitude);
}
}  // namespace detail

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template class InitializeGeometry<DIM(data)>;                                \
  template void detail::deriv_unnormalized_face_normals_impl(                  \
      gsl::not_null<DirectionMap<DIM(data), tnsr::ij<DataVector, DIM(data)>>*> \
          deriv_unnormalized_face_normals,                                     \
      const Mesh<DIM(data)>& mesh, const Element<DIM(data)>& element,          \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical,      \
                            Frame::Inertial>& inv_jacobian);                   \
  template tnsr::I<DataVector, DIM(data), Frame::ElementLogical>               \
  detail::mortar_logical_coordinates(                                          \
      const Mesh<DIM(data) - 1>& mortar_mesh,                                  \
      const ::dg::MortarSize<DIM(data) - 1>& mortar_size,                      \
      const Direction<DIM(data)>& direction);                                  \
  template void detail::mortar_jacobian(                                       \
      gsl::not_null<Scalar<DataVector>*> mortar_jacobian,                      \
      gsl::not_null<Scalar<DataVector>*> perpendicular_element_size,           \
      const Mesh<DIM(data) - 1>& mortar_mesh,                                  \
      const ::dg::MortarSize<DIM(data) - 1>& mortar_size,                      \
      const Direction<DIM(data)>& direction,                                   \
      const tnsr::I<DataVector, DIM(data), Frame::ElementLogical>&             \
          mortar_logical_coords,                                               \
      const std::optional<tnsr::II<DataVector, DIM(data)>>&                    \
          inv_metric_on_mortar,                                                \
      const ElementMap<DIM(data), Frame::Inertial>& element_map,               \
      const domain::FunctionsOfTimeMap& functions_of_time);                    \
  template void detail::initialize_coords_and_jacobians(                       \
      gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::ElementLogical>*>    \
          logical_coords,                                                      \
      gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>          \
          inertial_coords,                                                     \
      gsl::not_null<InverseJacobian<DataVector, DIM(data),                     \
                                    Frame::ElementLogical, Frame::Inertial>*>  \
          inv_jacobian,                                                        \
      gsl::not_null<Scalar<DataVector>*> det_inv_jacobian,                     \
      gsl::not_null<Scalar<DataVector>*> det_jacobian,                         \
      gsl::not_null<InverseJacobian<DataVector, DIM(data),                     \
                                    Frame::ElementLogical, Frame::Inertial>*>  \
          det_times_inv_jacobian,                                              \
      const Mesh<DIM(data)>& mesh,                                             \
      const ElementMap<DIM(data), Frame::Inertial>& element_map,               \
      const domain::FunctionsOfTimeMap& functions_of_time);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE

}  // namespace elliptic::dg
