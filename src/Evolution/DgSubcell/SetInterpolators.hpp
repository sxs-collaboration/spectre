// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <utility>

#include "DataStructures/ExtractPoint.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/BlockId.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/GhostZoneLogicalCoordinates.hpp"
#include "Evolution/DgSubcell/SliceTensor.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/Interpolators.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/Reconstructor.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::subcell {
/*!
 * \brief Sets the `intrp::IrregularInterpolant`s for interpolating to ghost
 * zone data at block boundaries.
 *
 * The DG to FD interpolants are at full order of the DG grid. The FD to FD
 * interpolant is piecewise linear with no support for neighboring
 * elements. We will want to use high-order slope-limited FD interpolation in
 * the future, but that requires neighbor communication. A slightly simpler
 * approach would be to use high-order Lagrange interpolation, which still
 * requires neighbor communication but does not require any additional changes
 * to the reconstruction routines to work on non-uniform grids. This is what
 * the Multipatch-MHD code does, relying on the slope limiting from the ghost
 * zones to remove oscillations. I (Nils Deppe) am not sure I love that, but
 * it's worth a try since it should be pretty easy to do.
 *
 * \warning Currently assumes that neighboring DG/FD elements are on the same
 * refinement level and have the same DG mesh and subcell mesh.
 */
template <size_t Dim>
struct SetInterpolators {
  using return_tags = tmpl::list<
      evolution::dg::subcell::Tags::InterpolatorsFromFdToNeighborFd<Dim>,
      evolution::dg::subcell::Tags::InterpolatorsFromDgToNeighborFd<Dim>,
      evolution::dg::subcell::Tags::InterpolatorsFromNeighborDgToFd<Dim>>;
  using argument_tags =
      tmpl::list<::domain::Tags::Element<Dim>, ::domain::Tags::Domain<Dim>,
                 domain::Tags::Mesh<Dim>, domain::Tags::Mesh<Dim>,
                 evolution::dg::subcell::Tags::Mesh<Dim>,
                 evolution::dg::subcell::Tags::Mesh<Dim>,
                 ::domain::Tags::ElementMap<Dim, Frame::Grid>,
                 evolution::dg::subcell::Tags::Reconstructor,
                 evolution::dg::subcell::Tags::SubcellOptions<Dim>>;

  template <typename ReconstructorType>
  static void apply(
      const gsl::not_null<
          DirectionalIdMap<Dim, std::optional<intrp::Irregular<Dim>>>*>
          interpolators_fd_to_neighbor_fd_ptr,
      const gsl::not_null<
          DirectionalIdMap<Dim, std::optional<intrp::Irregular<Dim>>>*>
          interpolators_dg_to_neighbor_fd_ptr,
      const gsl::not_null<
          DirectionalIdMap<Dim, std::optional<intrp::Irregular<Dim>>>*>
          interpolators_neighbor_dg_to_fd_ptr,
      const Element<Dim>& element, const Domain<Dim>& domain,
      const Mesh<Dim>& my_dg_mesh,
      // Needs to be updated to support non-uniform h/p-refinement
      const Mesh<Dim>& neighbor_dg_mesh, const Mesh<Dim>& my_fd_mesh,
      // Needs to be updated to support non-uniform h/p-refinement
      const Mesh<Dim>& neighbor_fd_mesh,
      const ElementMap<Dim, Frame::Grid>& element_map,
      const ReconstructorType& reconstructor,
      const evolution::dg::subcell::SubcellOptions& subcell_options) {
    if (alg::found(subcell_options.only_dg_block_ids(),
                   element.id().block_id())) {
      return;
    }

    const size_t number_of_ghost_zones = reconstructor.ghost_zone_size();
    const size_t my_block_id = element.id().block_id();
    for (const auto& direction_neighbors_in_direction : element.neighbors()) {
      const auto& direction = direction_neighbors_in_direction.first;
      const auto& neighbors_in_direction =
          direction_neighbors_in_direction.second;
      const auto& orientation = neighbors_in_direction.orientation();
      const auto direction_from_neighbor = orientation(direction.opposite());
      for (const ElementId<Dim>& neighbor_id : neighbors_in_direction) {
        const size_t neighbor_block_id = neighbor_id.block_id();
        if (neighbor_block_id == my_block_id) {
          continue;
        }
        const auto& neighbor_block = domain.blocks()[neighbor_block_id];
        // InterpolatorsFromFdToNeighborFd &
        // InterpolatorsFromDgToNeighborFd
        // 1. Compute the grid coordinates of my neighbor's ghost zones.
        // 2. Compute the element logical coordinates of my neighbor's
        //    ghost zones.
        // 3. Create interpolators

        if (not is_isotropic(neighbor_fd_mesh)) {
          ERROR("We assume an isotropic mesh but got "
                << neighbor_fd_mesh << " ElementID is " << element.id());
        }

        const auto get_logical_coords = [&element, &neighbor_id, &direction](
                                            const auto& map,
                                            const auto& grid_coords) {
          tnsr::I<DataVector, Dim, Frame::ElementLogical> logical_coords{
              get<0>(grid_coords).size()};
          for (size_t i = 0; i < get<0>(grid_coords).size(); ++i) {
            try {
              tnsr::I<double, Dim, Frame::ElementLogical> logical_coord =
                  map.inverse(extract_point(grid_coords, i));
              for (size_t d = 0; d < Dim; ++d) {
                logical_coords.get(d)[i] = logical_coord.get(d);
              }
            } catch (const std::bad_optional_access& e) {
              ERROR(
                  "Failed to get logical coordinates for neighbor's "
                  "ghost zone grid coordinates. This could be because the "
                  "ghost zones are not in the nearest neighbor but instead in "
                  "the next-to-nearest neighbor. The code assumes all ghost "
                  "zones, even on curved meshes, are in the nearest neighbors. "
                  "The current element is "
                  << element.id() << " and the neighbor id is " << neighbor_id
                  << " in direction " << direction
                  << " The neighbor grid coordinates are \n"
                  << extract_point(grid_coords, i) << "\n");
            }
          }
          return logical_coords;
        };

        tnsr::I<DataVector, Dim, Frame::Grid> neighbor_grid_ghost_zone_coords{};
        // Get the neighbor's ghost zone coordinates in the grid
        // frame.
        if (const tnsr::I<DataVector, Dim, Frame::ElementLogical>
                neighbor_logical_ghost_zone_coords =
                    evolution::dg::subcell::fd::ghost_zone_logical_coordinates(
                        neighbor_fd_mesh, number_of_ghost_zones,
                        direction_from_neighbor);
            neighbor_block.is_time_dependent()) {
          const ElementMap neighbor_element_map(
              neighbor_id,
              neighbor_block.moving_mesh_logical_to_grid_map().get_clone());
          neighbor_grid_ghost_zone_coords =
              neighbor_element_map(neighbor_logical_ghost_zone_coords);
        } else {
          const ElementMap neighbor_element_map(
              neighbor_id, neighbor_block.stationary_map().get_clone());
          const tnsr::I<DataVector, Dim, Frame::Inertial>
              neighbor_inertial_ghost_zone_coords =
                  neighbor_element_map(neighbor_logical_ghost_zone_coords);
          for (size_t i = 0; i < Dim; ++i) {
            neighbor_grid_ghost_zone_coords[i] =
                neighbor_inertial_ghost_zone_coords[i];
          }
        }
        // Map the ghost zone grid coordinates back to our logical
        // coordinates.
        const tnsr::I<DataVector, Dim, Frame::ElementLogical>
            neighbor_logical_ghost_zone_coords = get_logical_coords(
                element_map, neighbor_grid_ghost_zone_coords);

        // Set up interpolators for our local element to our neighbor's
        // ghost zones.
        (*interpolators_fd_to_neighbor_fd_ptr)[DirectionalId<Dim>{
            direction, neighbor_id}] = intrp::Irregular<Dim>{
            my_fd_mesh, neighbor_logical_ghost_zone_coords};
        (*interpolators_dg_to_neighbor_fd_ptr)[DirectionalId<Dim>{
            direction, neighbor_id}] = intrp::Irregular<Dim>{
            my_dg_mesh, neighbor_logical_ghost_zone_coords};

        // InterpolatorsFromNeighborDgToFd: the interpolation from our
        // neighbor's DG grid to our FD ghost zones.
        //
        // 1. Compute the grid coordinates of my ghost zones.
        // 2. Compute neighbor's element logical coordinates of my ghost
        //    zones
        // 3. Create interpolator for InterpolatorsFromNeighborDgToFd
        const tnsr::I<DataVector, Dim, Frame::ElementLogical>
            my_logical_coords = logical_coordinates(my_fd_mesh);
        tnsr::I<DataVector, Dim, Frame::ElementLogical>
            my_logical_ghost_zone_coords =
                evolution::dg::subcell::slice_tensor_for_subcell(
                    my_logical_coords, neighbor_fd_mesh.extents(),
                    number_of_ghost_zones, direction,
                    // We want to _set_ the interpolators, so just do a simple
                    // slice.
                    {});
        const double delta_xi =
            get<0>(my_logical_coords)[1] - get<0>(my_logical_coords)[0];
        // The sign accounts for whether we are shift along the
        // positive or negative axis.
        const double coordinate_shift =
            direction.sign() * delta_xi * number_of_ghost_zones;
        my_logical_ghost_zone_coords.get(direction.dimension()) +=
            coordinate_shift;
        const tnsr::I<DataVector, Dim, Frame::Grid> my_grid_ghost_zone_coords =
            element_map(my_logical_ghost_zone_coords);
        if (neighbor_block.is_time_dependent()) {
          const ElementMap neighbor_element_map(
              neighbor_id,
              neighbor_block.moving_mesh_logical_to_grid_map().get_clone());
          (*interpolators_neighbor_dg_to_fd_ptr)[DirectionalId<Dim>{
              direction, neighbor_id}] = intrp::Irregular<Dim>{
              neighbor_dg_mesh, get_logical_coords(neighbor_element_map,
                                                   my_grid_ghost_zone_coords)};
        } else {
          const ElementMap neighbor_element_map(
              neighbor_id, neighbor_block.stationary_map().get_clone());
          const tnsr::I<DataVector, Dim, Frame::Inertial>
              view_my_grid_ghost_zone_coords{};
          for (size_t i = 0; i < Dim; ++i) {
            make_const_view(make_not_null(&view_my_grid_ghost_zone_coords[i]),
                            my_grid_ghost_zone_coords[i], 0,
                            my_grid_ghost_zone_coords[i].size());
          }
          (*interpolators_neighbor_dg_to_fd_ptr)[DirectionalId<Dim>{
              direction, neighbor_id}] = intrp::Irregular<Dim>{
              neighbor_dg_mesh,
              get_logical_coords(neighbor_element_map,
                                 view_my_grid_ghost_zone_coords)};
        }
      }
    }
  }
};
}  // namespace evolution::dg::subcell
