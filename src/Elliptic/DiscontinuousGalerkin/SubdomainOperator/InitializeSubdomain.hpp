// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/OverlapHelpers.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}  // namespace Parallel
namespace tuples {
template <typename... Tags>
struct TaggedTuple;
}  // namespace tuples
/// \endcond

namespace elliptic::dg::Actions {

/*!
 * \brief Initialize the geometry for the DG subdomain operator
 *
 * Initializes tags that define the geometry of overlap regions with neighboring
 * elements. The data needs to be updated if the geometry of neighboring
 * elements changes.
 */
template <size_t Dim, typename OptionsGroup>
struct InitializeSubdomain {
 private:
  template <typename Tag>
  using overlaps_tag =
      LinearSolver::Schwarz::Tags::Overlaps<Tag, Dim, OptionsGroup>;
  template <typename ValueType>
  using overlaps = LinearSolver::Schwarz::OverlapMap<Dim, ValueType>;

 public:
  using initialization_tags =
      tmpl::list<domain::Tags::InitialExtents<Dim>,
                 domain::Tags::InitialRefinementLevels<Dim>>;
  using const_global_cache_tags =
      tmpl::list<LinearSolver::Schwarz::Tags::MaxOverlap<OptionsGroup>>;
  using simple_tags = db::wrap_tags_in<
      overlaps_tag,
      tmpl::list<
          domain::Tags::Mesh<Dim>,
          elliptic::dg::subdomain_operator::Tags::ExtrudingExtent,
          domain::Tags::Element<Dim>, domain::Tags::ElementMap<Dim>,
          domain::Tags::InverseJacobian<Dim, Frame::Logical, Frame::Inertial>,
          domain::Tags::Interface<
              domain::Tags::InternalDirections<Dim>,
              ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>,
          domain::Tags::Interface<
              domain::Tags::BoundaryDirectionsInterior<Dim>,
              ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>,
          domain::Tags::Interface<
              domain::Tags::InternalDirections<Dim>,
              ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>>,
          domain::Tags::Interface<
              domain::Tags::BoundaryDirectionsInterior<Dim>,
              ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>>,
          ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>,
          ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>,
          elliptic::dg::subdomain_operator::Tags::NeighborMortars<
              domain::Tags::Mesh<Dim>, Dim>,
          elliptic::dg::subdomain_operator::Tags::NeighborMortars<
              ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>,
              Dim>,
          elliptic::dg::subdomain_operator::Tags::NeighborMortars<
              domain::Tags::Mesh<Dim - 1>, Dim>,
          elliptic::dg::subdomain_operator::Tags::NeighborMortars<
              ::Tags::MortarSize<Dim - 1>, Dim>>>;
  using compute_tags = tmpl::list<>;

  template <
      typename DataBox, typename... InboxTags, typename Metavariables,
      typename ActionList, typename ParallelComponent,
      Requires<tmpl::all<initialization_tags,
                         tmpl::bind<db::tag_is_retrievable, tmpl::_1,
                                    tmpl::pin<DataBox>>>::value> = nullptr>
  static std::tuple<DataBox&&> apply(
      DataBox& box, const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*element_id*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& initial_extents =
        db::get<domain::Tags::InitialExtents<Dim>>(box);
    const auto& initial_refinement =
        db::get<domain::Tags::InitialRefinementLevels<Dim>>(box);
    const auto& domain = db::get<domain::Tags::Domain<Dim>>(box);
    const auto& max_overlap =
        get<LinearSolver::Schwarz::Tags::MaxOverlap<OptionsGroup>>(box);

    overlaps<Mesh<Dim>> overlap_meshes{};
    overlaps<size_t> overlap_extents{};
    overlaps<Element<Dim>> overlap_elements{};
    overlaps<ElementMap<Dim, Frame::Inertial>> overlap_element_maps{};
    overlaps<InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>>
        overlap_inv_jacobians{};
    overlaps<std::unordered_map<Direction<Dim>, tnsr::i<DataVector, Dim>>>
        overlap_face_normals_internal{};
    overlaps<std::unordered_map<Direction<Dim>, tnsr::i<DataVector, Dim>>>
        overlap_face_normals_external{};
    overlaps<std::unordered_map<Direction<Dim>, Scalar<DataVector>>>
        overlap_face_normal_magnitudes_internal{};
    overlaps<std::unordered_map<Direction<Dim>, Scalar<DataVector>>>
        overlap_face_normal_magnitudes_external{};
    overlaps<::dg::MortarMap<Dim, Mesh<Dim - 1>>> overlap_mortar_meshes{};
    overlaps<::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>>>
        overlap_mortar_sizes{};
    overlaps<::dg::MortarMap<Dim, Mesh<Dim>>> overlap_neighbor_meshes{};
    overlaps<::dg::MortarMap<Dim, Scalar<DataVector>>>
        overlap_neighbor_face_normal_magnitudes{};
    overlaps<::dg::MortarMap<Dim, Mesh<Dim - 1>>>
        overlap_neighbor_mortar_meshes{};
    overlaps<::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>>>
        overlap_neighbor_mortar_sizes{};

    const auto& element = db::get<domain::Tags::Element<Dim>>(box);
    for (const auto& [direction, neighbors] : element.neighbors()) {
      const auto& orientation = neighbors.orientation();
      const auto& direction_from_neighbor = orientation(direction.opposite());
      const auto& dimension_in_neighbor = direction_from_neighbor.dimension();
      for (const auto& neighbor_id : neighbors) {
        const auto overlap_id = std::make_pair(direction, neighbor_id);
        // Mesh
        overlap_meshes.emplace(overlap_id,
                               domain::Initialization::create_initial_mesh(
                                   initial_extents, neighbor_id,
                                   Spectral::Quadrature::GaussLobatto));
        const auto& neighbor_mesh = overlap_meshes.at(overlap_id);
        // Overlap extents
        overlap_extents.emplace(
            overlap_id,
            LinearSolver::Schwarz::overlap_extent(
                neighbor_mesh.extents(dimension_in_neighbor), max_overlap));
        // Element
        const auto& neighbor_block = domain.blocks()[neighbor_id.block_id()];
        overlap_elements.emplace(
            overlap_id, domain::Initialization::create_initial_element(
                            neighbor_id, neighbor_block, initial_refinement));
        const auto& neighbor = overlap_elements.at(overlap_id);
        // Element map
        overlap_element_maps.emplace(
            overlap_id,
            ElementMap<Dim, Frame::Inertial>{
                neighbor_id, neighbor_block.stationary_map().get_clone()});
        const auto& neighbor_element_map = overlap_element_maps.at(overlap_id);
        // Jacobian
        const auto neighbor_logical_coords = logical_coordinates(neighbor_mesh);
        overlap_inv_jacobians.emplace(
            overlap_id,
            neighbor_element_map.inv_jacobian(neighbor_logical_coords));
        // Faces and mortars, internal and external
        std::unordered_map<Direction<Dim>, tnsr::i<DataVector, Dim>>
            neighbor_face_normals_internal{};
        std::unordered_map<Direction<Dim>, tnsr::i<DataVector, Dim>>
            neighbor_face_normals_external{};
        std::unordered_map<Direction<Dim>, Scalar<DataVector>>
            neighbor_face_normal_magnitudes_internal{};
        std::unordered_map<Direction<Dim>, Scalar<DataVector>>
            neighbor_face_normal_magnitudes_external{};
        const auto setup_face =
            [&neighbor_face_normals_internal, &neighbor_face_normals_external,
             &neighbor_face_normal_magnitudes_internal,
             &neighbor_face_normal_magnitudes_external, &neighbor_mesh,
             &neighbor_element_map](const Direction<Dim>& local_direction,
                                    const bool is_external) {
              auto& neighbor_face_normals =
                  is_external ? neighbor_face_normals_external
                              : neighbor_face_normals_internal;
              auto& neighbor_face_normal_magnitudes =
                  is_external ? neighbor_face_normal_magnitudes_external
                              : neighbor_face_normal_magnitudes_internal;
              const auto neighbor_face_mesh =
                  neighbor_mesh.slice_away(local_direction.dimension());
              auto neighbor_face_normal = unnormalized_face_normal(
                  neighbor_face_mesh, neighbor_element_map, local_direction);
              Scalar<DataVector> neighbor_normal_magnitude{
                  neighbor_face_mesh.number_of_grid_points()};
              magnitude(make_not_null(&neighbor_normal_magnitude),
                        neighbor_face_normal);
              for (size_t d = 0; d < Dim; d++) {
                neighbor_face_normal.get(d) /= get(neighbor_normal_magnitude);
              }
              neighbor_face_normals.emplace(local_direction,
                                            std::move(neighbor_face_normal));
              neighbor_face_normal_magnitudes.emplace(
                  local_direction, std::move(neighbor_normal_magnitude));
            };
        ::dg::MortarMap<Dim, Mesh<Dim - 1>> neighbor_mortar_meshes{};
        ::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>> neighbor_mortar_sizes{};
        for (const auto& [neighbor_direction, neighbor_neighbors] :
             neighbor.neighbors()) {
          setup_face(neighbor_direction, false);
          const auto neighbor_dimension = neighbor_direction.dimension();
          const auto neighbor_face_mesh =
              neighbor_mesh.slice_away(neighbor_dimension);
          for (const auto& neighbor_neighbor_id : neighbor_neighbors) {
            const auto neighbor_mortar_id =
                std::make_pair(neighbor_direction, neighbor_neighbor_id);
            neighbor_mortar_meshes.emplace(
                neighbor_mortar_id,
                ::dg::mortar_mesh(neighbor_face_mesh,
                                  domain::Initialization::create_initial_mesh(
                                      initial_extents, neighbor_neighbor_id,
                                      Spectral::Quadrature::GaussLobatto,
                                      neighbor_neighbors.orientation())
                                      .slice_away(neighbor_dimension)));
            neighbor_mortar_sizes.emplace(
                neighbor_mortar_id,
                ::dg::mortar_size(neighbor_id, neighbor_neighbor_id,
                                  neighbor_dimension,
                                  neighbor_neighbors.orientation()));
          }
        }
        for (const auto& neighbor_direction : neighbor.external_boundaries()) {
          setup_face(neighbor_direction, true);
          const auto neighbor_mortar_id = std::make_pair(
              neighbor_direction, ElementId<Dim>::external_boundary_id());
          neighbor_mortar_meshes.emplace(
              neighbor_mortar_id,
              neighbor_mesh.slice_away(neighbor_direction.dimension()));
          neighbor_mortar_sizes.emplace(
              neighbor_mortar_id,
              make_array<Dim - 1>(Spectral::MortarSize::Full));
        }
        overlap_face_normals_internal.emplace(
            overlap_id, std::move(neighbor_face_normals_internal));
        overlap_face_normals_external.emplace(
            overlap_id, std::move(neighbor_face_normals_external));
        overlap_face_normal_magnitudes_internal.emplace(
            overlap_id, std::move(neighbor_face_normal_magnitudes_internal));
        overlap_face_normal_magnitudes_external.emplace(
            overlap_id, std::move(neighbor_face_normal_magnitudes_external));
        overlap_mortar_meshes.emplace(overlap_id,
                                      std::move(neighbor_mortar_meshes));
        overlap_mortar_sizes.emplace(overlap_id,
                                     std::move(neighbor_mortar_sizes));

        // Neighbor's neighbors
        ::dg::MortarMap<Dim, Mesh<Dim>> neighbors_neighbor_meshes{};
        ::dg::MortarMap<Dim, Scalar<DataVector>>
            neighbors_neighbor_face_normal_magnitudes{};
        ::dg::MortarMap<Dim, Mesh<Dim - 1>> neighbors_neighbor_mortar_meshes{};
        ::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>>
            neighbors_neighbor_mortar_sizes{};
        for (const auto& [neighbor_direction, neighbor_neighbors] :
             neighbor.neighbors()) {
          const auto& neighbors_neighbor_orientation =
              neighbor_neighbors.orientation();
          const auto direction_from_neighbors_neighbor =
              neighbors_neighbor_orientation(neighbor_direction.opposite());
          const auto reoriented_neighbor_face_mesh =
              neighbors_neighbor_orientation(neighbor_mesh)
                  .slice_away(direction_from_neighbors_neighbor.dimension());
          for (const auto& neighbors_neighbor_id : neighbor_neighbors) {
            const auto neighbors_neighbor_mortar_id =
                std::make_pair(neighbor_direction, neighbors_neighbor_id);
            neighbors_neighbor_meshes.emplace(
                neighbors_neighbor_mortar_id,
                domain::Initialization::create_initial_mesh(
                    initial_extents, neighbors_neighbor_id,
                    Spectral::Quadrature::GaussLobatto));
            const auto& neighbors_neighbor_mesh =
                neighbors_neighbor_meshes.at(neighbors_neighbor_mortar_id);
            const auto neighbors_neighbor_face_mesh =
                neighbors_neighbor_mesh.slice_away(
                    direction_from_neighbors_neighbor.dimension());
            const auto& neighbors_neighbor_block =
                domain.blocks()[neighbors_neighbor_id.block_id()];
            ElementMap<Dim, Frame::Inertial> neighbors_neighbor_element_map{
                neighbors_neighbor_id,
                neighbors_neighbor_block.stationary_map().get_clone()};
            const auto neighbors_neighbor_face_normal =
                unnormalized_face_normal(neighbors_neighbor_face_mesh,
                                         neighbors_neighbor_element_map,
                                         direction_from_neighbors_neighbor);
            Scalar<DataVector> neighbors_neighbor_face_normal_magnitude{
                neighbors_neighbor_face_mesh.number_of_grid_points()};
            magnitude(make_not_null(&neighbors_neighbor_face_normal_magnitude),
                      neighbors_neighbor_face_normal);
            neighbors_neighbor_face_normal_magnitudes.emplace(
                neighbors_neighbor_mortar_id,
                std::move(neighbors_neighbor_face_normal_magnitude));
            neighbors_neighbor_mortar_meshes.emplace(
                neighbors_neighbor_mortar_id,
                ::dg::mortar_mesh(reoriented_neighbor_face_mesh,
                                  neighbors_neighbor_face_mesh));
            neighbors_neighbor_mortar_sizes.emplace(
                neighbors_neighbor_mortar_id,
                ::dg::mortar_size(
                    neighbors_neighbor_id, neighbor_id,
                    direction_from_neighbors_neighbor.dimension(),
                    neighbors_neighbor_orientation.inverse_map()));
          }
        }
        overlap_neighbor_meshes.emplace(overlap_id,
                                        std::move(neighbors_neighbor_meshes));
        overlap_neighbor_face_normal_magnitudes.emplace(
            overlap_id, std::move(neighbors_neighbor_face_normal_magnitudes));
        overlap_neighbor_mortar_meshes.emplace(
            overlap_id, std::move(neighbors_neighbor_mortar_meshes));
        overlap_neighbor_mortar_sizes.emplace(
            overlap_id, std::move(neighbors_neighbor_mortar_sizes));
      }  // neighbors in direction
    }    // directions

    ::Initialization::mutate_assign<simple_tags>(
        make_not_null(&box), std::move(overlap_meshes),
        std::move(overlap_extents), std::move(overlap_elements),
        std::move(overlap_element_maps), std::move(overlap_inv_jacobians),
        std::move(overlap_face_normals_internal),
        std::move(overlap_face_normals_external),
        std::move(overlap_face_normal_magnitudes_internal),
        std::move(overlap_face_normal_magnitudes_external),
        std::move(overlap_mortar_meshes), std::move(overlap_mortar_sizes),
        std::move(overlap_neighbor_meshes),
        std::move(overlap_neighbor_face_normal_magnitudes),
        std::move(overlap_neighbor_mortar_meshes),
        std::move(overlap_neighbor_mortar_sizes));
    return {std::move(box)};
  }

  template <
      typename DataBox, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<not tmpl::all<initialization_tags,
                             tmpl::bind<db::tag_is_retrievable, tmpl::_1,
                                        tmpl::pin<DataBox>>>::value> = nullptr>
  static std::tuple<DataBox&&> apply(
      DataBox& /*box*/, const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    ERROR(
        "Dependencies not fulfilled. Did you forget to terminate the phase "
        "after removing options?");
  }
};

}  // namespace elliptic::dg::Actions
