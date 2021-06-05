// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/DiscontinuousGalerkin/Initialization.hpp"

#include <array>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
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
    const gsl::not_null<std::unordered_set<Direction<Dim>>*>
        internal_directions,
    const gsl::not_null<std::unordered_set<Direction<Dim>>*>
        external_directions,
    const gsl::not_null<std::unordered_map<Direction<Dim>, Direction<Dim>>*>
        face_directions_internal,
    const gsl::not_null<
        std::unordered_map<Direction<Dim>, tnsr::I<DataVector, Dim>>*>
        face_inertial_coords_internal,
    const gsl::not_null<
        std::unordered_map<Direction<Dim>, tnsr::i<DataVector, Dim>>*>
        face_normals_internal,
    const gsl::not_null<std::unordered_map<Direction<Dim>, Scalar<DataVector>>*>
    /*face_normal_magnitudes_internal*/,
    const gsl::not_null<std::unordered_map<Direction<Dim>, Direction<Dim>>*>
        face_directions_external,
    const gsl::not_null<
        std::unordered_map<Direction<Dim>, tnsr::I<DataVector, Dim>>*>
        face_inertial_coords_external,
    const gsl::not_null<
        std::unordered_map<Direction<Dim>, tnsr::i<DataVector, Dim>>*>
        face_normals_external,
    const gsl::not_null<std::unordered_map<Direction<Dim>, Scalar<DataVector>>*>
    /*face_normal_magnitudes_external*/,
    const gsl::not_null<::dg::MortarMap<Dim, Mesh<Dim - 1>>*> mortar_meshes,
    const gsl::not_null<::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>>*>
        mortar_sizes,
    const Mesh<Dim>& mesh, const Element<Dim>& element,
    const ElementMap<Dim, Frame::Inertial>& element_map,
    const std::vector<std::array<size_t, Dim>>& initial_extents)
    const noexcept {
  const Spectral::Quadrature quadrature = mesh.quadrature(0);
  const auto& element_id = element.id();
  *internal_directions = element.internal_boundaries();
  *external_directions = element.external_boundaries();
  // Internal faces and mortars
  for (const auto& [direction, neighbors] : element.neighbors()) {
    const auto face_mesh = mesh.slice_away(direction.dimension());
    (*face_directions_internal)[direction] = direction;
    // Possible optimization: Not all systems need the coordinates on internal
    // faces.
    (*face_inertial_coords_internal)[direction] = element_map.operator()(
        interface_logical_coordinates(face_mesh, direction));
    (*face_normals_internal)[direction] =
        unnormalized_face_normal(face_mesh, element_map, direction);
    const auto& orientation = neighbors.orientation();
    for (const auto& neighbor_id : neighbors) {
      const ::dg::MortarId<Dim> mortar_id{direction, neighbor_id};
      // Geometry on this side of the mortar
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
  // External faces and mortars
  for (const auto& direction : element.external_boundaries()) {
    const auto face_mesh = mesh.slice_away(direction.dimension());
    (*face_directions_external)[direction] = direction;
    (*face_inertial_coords_external)[direction] = element_map.operator()(
        interface_logical_coordinates(face_mesh, direction));
    (*face_normals_external)[direction] =
        unnormalized_face_normal(face_mesh, element_map, direction);
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
