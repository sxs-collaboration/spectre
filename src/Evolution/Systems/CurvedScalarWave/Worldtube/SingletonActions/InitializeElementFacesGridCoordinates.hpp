// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/InitialExtents.hpp"
#include "Domain/Creators/Tags/InitialRefinementLevels.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/ExcisionSphere.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace CurvedScalarWave::Worldtube::Initialization {

/*!
 * \brief Initializes a map of the grid coordinates centered on the worldtube of
 * all faces that abut the worldtube with corresponding ElementIds.
 *
 * \details The worldtube singleton computes an internal solution in the grid
 * frame and uses this map to evaluate compute it at the grid coordinates of
 * each element face abutting the worldtube each time step. This data is sent to
 * the corresponding element where it is used to apply pointwise boundary
 * conditions. The `ElementFacesGridCoordinates` holds a map of all the element
 * ids abutting the worldtube with the corresonding grid coordinates.
 *
 * \warning This currently assumes that initial domain remains the same and
 * there is no AMR. To support this, the worldtube could send the
 * coefficients of its internal solution to each element which can evaluate
 * it on their current grid in the boundary conditions.
 *
 *  DataBox changes:
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies: Tags::ElementFacesCoordinatesMap<Dim>
 */
template <size_t Dim>
struct InitializeElementFacesGridCoordinates {
  using return_tags = tmpl::list<Tags::ElementFacesGridCoordinates<Dim>>;
  using simple_tags = tmpl::list<Tags::ElementFacesGridCoordinates<Dim>>;
  using compute_tags = tmpl::list<>;
  using simple_tags_from_options =
      tmpl::list<::domain::Tags::InitialExtents<Dim>,
                 ::domain::Tags::InitialRefinementLevels<Dim>,
                 evolution::dg::Tags::Quadrature>;
  using const_global_cache_tags = tmpl::list<>;
  using mutable_global_cache_tags = tmpl::list<>;
  using argument_tags = tmpl::flatten<
      tmpl::list<simple_tags_from_options, ::domain::Tags::Domain<Dim>,
                 Tags::ExcisionSphere<Dim>>>;
  static void apply(
      const gsl::not_null<std::unordered_map<
          ElementId<Dim>, tnsr::I<DataVector, Dim, Frame::Grid>>*>
          element_faces_grid_coords,
      const std::vector<std::array<size_t, Dim>>& initial_extents,
      const std::vector<std::array<size_t, Dim>>& initial_refinement,
      const Spectral::Quadrature& quadrature, const Domain<Dim>& domain,
      const ::ExcisionSphere<Dim>& excision_sphere) {
    const auto& blocks = domain.blocks();
    const auto& worldtube_grid_coords = excision_sphere.center();
    const auto& neighboring_blocks = excision_sphere.abutting_directions();
    for (const auto& [block_id, _] : neighboring_blocks) {
      const auto element_ids =
          initial_element_ids(block_id, initial_refinement.at(block_id));
      for (const auto element_id : element_ids) {
        const auto direction = excision_sphere.abutting_direction(element_id);
        if (direction.has_value()) {
          const auto mesh = ::domain::Initialization::create_initial_mesh(
              initial_extents, element_id, quadrature);
          const auto face_mesh = mesh.slice_away(direction.value().dimension());
          const auto& current_block = blocks.at(block_id);
          const ElementMap<Dim, Frame::Grid> element_map{
              element_id,
              current_block.is_time_dependent()
                  ? current_block.moving_mesh_logical_to_grid_map().get_clone()
                  : current_block.stationary_map().get_to_grid_frame()};
          const auto face_logical_coords =
              interface_logical_coordinates(face_mesh, direction.value());
          auto faces_grid_coords = element_map(face_logical_coords);
          for (size_t i = 0; i < 3; ++i) {
            faces_grid_coords.get(i) -= worldtube_grid_coords.get(i);
          }
          element_faces_grid_coords->operator[](element_id) =
              std::move(faces_grid_coords);
        }
      }
    }
  }
};
}  // namespace CurvedScalarWave::Worldtube::Initialization
