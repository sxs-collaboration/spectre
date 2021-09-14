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
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/IndexToSliceAt.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace elliptic::dg {

template <size_t Dim>
void InitializeGeometry<Dim>::operator()(
    const gsl::not_null<Mesh<Dim>*> mesh,
    const gsl::not_null<Element<Dim>*> element,
    const gsl::not_null<ElementMap<Dim, Frame::Inertial>*> element_map,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Logical>*>
        logical_coords,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        inertial_coords,
    const gsl::not_null<
        InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>*>
        inv_jacobian,
    const gsl::not_null<Scalar<DataVector>*> det_inv_jacobian,
    const std::vector<std::array<size_t, Dim>>& initial_extents,
    const std::vector<std::array<size_t, Dim>>& initial_refinement,
    const Domain<Dim>& domain,
    const ElementId<Dim>& element_id) const noexcept {
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

template <size_t Dim>
void InitializeFacesAndMortars<Dim>::operator()(
    const gsl::not_null<DirectionMap<Dim, Direction<Dim>>*> face_directions,
    const gsl::not_null<DirectionMap<Dim, tnsr::I<DataVector, Dim>>*>
        face_inertial_coords,
    const gsl::not_null<DirectionMap<Dim, tnsr::i<DataVector, Dim>>*>
        face_normals,
    const gsl::not_null<DirectionMap<Dim, Scalar<DataVector>>*>
    /*face_normal_magnitudes*/,
    const gsl::not_null<DirectionMap<Dim, tnsr::ij<DataVector, Dim>>*>
        deriv_unnormalized_face_normals,
    const gsl::not_null<::dg::MortarMap<Dim, Mesh<Dim - 1>>*> mortar_meshes,
    const gsl::not_null<::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>>*>
        mortar_sizes,
    const Mesh<Dim>& mesh, const Element<Dim>& element,
    const ElementMap<Dim, Frame::Inertial>& element_map,
    const InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>&
        inv_jacobian,
    const std::vector<std::array<size_t, Dim>>& initial_extents)
    const noexcept {
  const Spectral::Quadrature quadrature = mesh.quadrature(0);
  // Faces
  for (const auto& direction : Direction<Dim>::all_directions()) {
    const auto face_mesh = mesh.slice_away(direction.dimension());
    (*face_directions)[direction] = direction;
    // Possible optimization: Not all systems need the coordinates on internal
    // faces.
    (*face_inertial_coords)[direction] = element_map.operator()(
        interface_logical_coordinates(face_mesh, direction));
    (*face_normals)[direction] =
        unnormalized_face_normal(face_mesh, element_map, direction);
  }
  // Compute the Jacobian derivative numerically, because our coordinate maps
  // currently don't provide it analytically.
  if (not element.external_boundaries().empty()) {
    ASSERT(mesh.quadrature(0) == Spectral::Quadrature::GaussLobatto,
           "Slicing the Hessian to the boundary currently supports only "
           "Gauss-Lobatto grids. Add support to "
           "'elliptic::dg::InitializeFacesAndMortars'.");
    using inv_jac_tag =
        domain::Tags::InverseJacobian<Dim, Frame::Logical, Frame::Inertial>;
    Variables<tmpl::list<inv_jac_tag>> vars_to_differentiate{
        mesh.number_of_grid_points()};
    get<inv_jac_tag>(vars_to_differentiate) = inv_jacobian;
    const auto deriv_vars = partial_derivatives<tmpl::list<inv_jac_tag>>(
        vars_to_differentiate, mesh, inv_jacobian);
    const auto& deriv_inv_jac =
        get<::Tags::deriv<inv_jac_tag, tmpl::size_t<Dim>, Frame::Inertial>>(
            deriv_vars);
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
  // Mortars
  const auto& element_id = element.id();
  for (const auto& [direction, neighbors] : element.neighbors()) {
    const auto face_mesh = mesh.slice_away(direction.dimension());
    const auto& orientation = neighbors.orientation();
    for (const auto& neighbor_id : neighbors) {
      const ::dg::MortarId<Dim> mortar_id{direction, neighbor_id};
      mortar_meshes->emplace(
          mortar_id, ::dg::mortar_mesh(
                         face_mesh, domain::Initialization::create_initial_mesh(
                                        initial_extents, neighbor_id,
                                        quadrature, orientation)
                                        .slice_away(direction.dimension())));
      mortar_sizes->emplace(
          mortar_id, ::dg::mortar_size(element_id, neighbor_id,
                                       direction.dimension(), orientation));
    }  // neighbors
  }    // internal directions
  for (const auto& direction : element.external_boundaries()) {
    const auto face_mesh = mesh.slice_away(direction.dimension());
    const auto mortar_id =
        std::make_pair(direction, ElementId<Dim>::external_boundary_id());
    mortar_meshes->emplace(mortar_id, face_mesh);
    mortar_sizes->emplace(mortar_id,
                          make_array<Dim - 1>(Spectral::MortarSize::Full));
  }  // external directions
}

template class InitializeGeometry<1>;
template class InitializeGeometry<2>;
template class InitializeGeometry<3>;
template class InitializeFacesAndMortars<1>;
template class InitializeFacesAndMortars<2>;
template class InitializeFacesAndMortars<3>;

}  // namespace elliptic::dg
