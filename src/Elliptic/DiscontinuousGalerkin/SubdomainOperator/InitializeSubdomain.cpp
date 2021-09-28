// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/InitializeSubdomain.hpp"

#include <array>
#include <cstddef>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/OverlapHelpers.hpp"
#include "Utilities/Gsl.hpp"

namespace elliptic::dg::subdomain_operator::Actions::detail {

template <size_t Dim>
void InitializeOverlapGeometry<Dim>::operator()(
    const gsl::not_null<size_t*> extruding_extent,
    const gsl::not_null<::dg::MortarMap<Dim, Mesh<Dim>>*> neighbor_meshes,
    const gsl::not_null<::dg::MortarMap<
        Dim, Scalar<DataVector>>*> /*neighbor_face_normal_magnitudes*/,
    const gsl::not_null<::dg::MortarMap<Dim, Mesh<Dim - 1>>*>
        neighbor_mortar_meshes,
    const gsl::not_null<::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>>*>
        neighbor_mortar_sizes,
    const Element<Dim>& element, const Mesh<Dim>& mesh,
    const std::vector<std::array<size_t, Dim>>& initial_extents,
    const ElementId<Dim>& element_id, const Direction<Dim>& overlap_direction,
    const size_t max_overlap) const {
  const Spectral::Quadrature quadrature = mesh.quadrature(0);
  // Extruding extent
  *extruding_extent = LinearSolver::Schwarz::overlap_extent(
      mesh.extents(overlap_direction.dimension()), max_overlap);
  // Geometry on the remote side of the mortar. These are only needed on mortars
  // to neighbors that are not part of the subdomain, so conditionally skipping
  // this setup is a possible optimization. The computational cost and memory
  // usage is probably irrelevant though.
  for (const auto& [direction, neighbors] : element.neighbors()) {
    const auto& orientation = neighbors.orientation();
    const auto direction_from_neighbor = orientation(direction.opposite());
    const auto reoriented_face_mesh =
        orientation(mesh).slice_away(direction_from_neighbor.dimension());
    for (const auto& neighbor_id : neighbors) {
      const ::dg::MortarId<Dim> mortar_id{direction, neighbor_id};
      const auto& neighbor_mesh =
          neighbor_meshes
              ->emplace(mortar_id,
                        domain::Initialization::create_initial_mesh(
                            initial_extents, neighbor_id, quadrature))
              .first->second;
      const auto neighbor_face_mesh =
          neighbor_mesh.slice_away(direction_from_neighbor.dimension());
      neighbor_mortar_meshes->emplace(
          mortar_id,
          ::dg::mortar_mesh(reoriented_face_mesh, neighbor_face_mesh));
      neighbor_mortar_sizes->emplace(
          mortar_id, ::dg::mortar_size(neighbor_id, element_id,
                                       direction_from_neighbor.dimension(),
                                       orientation.inverse_map()));
    }  // neighbors
  }    // internal directions
}

template class InitializeOverlapGeometry<1>;
template class InitializeOverlapGeometry<2>;
template class InitializeOverlapGeometry<3>;

}  // namespace elliptic::dg::subdomain_operator::Actions::detail
