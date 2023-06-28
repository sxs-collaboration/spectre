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
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace elliptic::dg {

template <size_t Dim>
void InitializeGeometry<Dim>::operator()(
    const gsl::not_null<Mesh<Dim>*> mesh,
    const gsl::not_null<Element<Dim>*> element,
    const gsl::not_null<ElementMap<Dim, Frame::Inertial>*> element_map,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::ElementLogical>*>
        logical_coords,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        inertial_coords,
    const gsl::not_null<InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                                        Frame::Inertial>*>
        inv_jacobian,
    const gsl::not_null<Scalar<DataVector>*> det_inv_jacobian,
    const std::vector<std::array<size_t, Dim>>& initial_extents,
    const std::vector<std::array<size_t, Dim>>& initial_refinement,
    const Domain<Dim>& domain, const ElementId<Dim>& element_id) const {
  // Mesh
  const auto quadrature = Spectral::Quadrature::GaussLobatto;
  *mesh = domain::Initialization::create_initial_mesh(initial_extents,
                                                      element_id, quadrature);
  // Element
  const auto& block = domain.blocks()[element_id.block_id()];
  if (block.is_time_dependent()) {
    ERROR_NO_TRACE(
        "The InitializeDomain action is for elliptic systems which do not have "
        "any time-dependence, but the domain creator has set up the domain to "
        "have time-dependence.");
  }
  *element = domain::Initialization::create_initial_element(element_id, block,
                                                            initial_refinement);
  // Element map
  *element_map = ElementMap<Dim, Frame::Inertial>{
      element_id, block.stationary_map().get_clone()};
  // Coordinates
  *logical_coords = logical_coordinates(*mesh);
  *inertial_coords = element_map->operator()(*logical_coords);
  // Jacobian
  *inv_jacobian = element_map->inv_jacobian(*logical_coords);
  *det_inv_jacobian = determinant(*inv_jacobian);
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
  ASSERT(mesh.quadrature(0) == Spectral::Quadrature::GaussLobatto,
         "Slicing the Hessian to the boundary currently supports only "
         "Gauss-Lobatto grids. Add support to "
         "'elliptic::dg::InitializeFacesAndMortars'.");
  const auto deriv_inv_jac =
      partial_derivative(inv_jacobian, mesh, inv_jacobian);
  for (const auto& direction : element.external_boundaries()) {
    const auto deriv_inv_jac_on_face =
        data_on_slice(deriv_inv_jac, mesh.extents(), direction.dimension(),
                      index_to_slice_at(mesh.extents(), direction));
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
      const Direction<DIM(data)>& direction);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE

}  // namespace elliptic::dg
