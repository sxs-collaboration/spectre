// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/ElementMap.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/GhostZoneLogicalCoordinates.hpp"
#include "Evolution/DgSubcell/Tags/GhostZoneInverseJacobian.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::subcell {

/// \brief Mutator that stores the grid coordinates and inverse Jacobians of the
/// ghost zone.
///
/// \details Mutator that stores the grid coordinates and inverse Jacobians of
/// the ghost zone. This is run in the initialization phase since this
/// information is time-independent. The full Jacobian to inertial coordinates
/// may be applied at each time step.

template <size_t Dim, typename ReconstructorTag>
struct GhostZoneInverseJacobian {
  /// Tags for constant items added to the GlobalCache.  These items are
  /// initialized from input file options.
  using const_global_cache_tags = tmpl::list<>;

  /// Tags for mutable items added to the GlobalCache.  These items are
  /// initialized from input file options.
  using mutable_global_cache_tags = tmpl::list<>;

  /// Tags for simple DataBox items that are initialized from input file options
  using simple_tags_from_options = tmpl::list<>;

  /// Tags for simple DataBox items that are default initialized.
  using default_initialized_simple_tags = tmpl::list<>;

  /// Tags for items fetched by the DataBox and passed to the apply function
  using argument_tags =
      tmpl::list<Tags::Mesh<Dim>, ::domain::Tags::ElementMap<Dim, Frame::Grid>,
                 ReconstructorTag>;

  /// Tags for items in the DataBox that are mutated by the apply function
  using return_tags = tmpl::list<Tags::GhostZoneInverseJacobian<Dim>>;

  /// Tags for mutable DataBox items that are either default initialized or
  /// initialized by the apply function
  using simple_tags = return_tags;

  /// Tags for immutable DataBox items (compute items or reference items) added
  /// to the DataBox.
  using compute_tags = tmpl::list<>;

  /// Given the items fetched from a DataBox by the argument_tags, mutate
  /// the items in the DataBox corresponding to return_tags
  template <typename ReconstructorType>
  static void apply(
      const gsl::not_null<DirectionMap<
          Dim, Variables<tmpl::list<
                   evolution::dg::subcell::Tags::Coordinates<Dim, Frame::Grid>,
                   evolution::dg::subcell::fd::Tags::
                       InverseJacobianLogicalToGrid<Dim>>>>*>
          ghost_zone_inverse_jacobian,
      const Mesh<Dim>& subcell_mesh,
      const ElementMap<Dim, Frame::Grid>& element_map,
      const ReconstructorType& reconstructor) {
    using neighbor_tags = tmpl::list<
        evolution::dg::subcell::Tags::Coordinates<Dim, Frame::Grid>,
        evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToGrid<Dim>>;

    for (const auto& direction : Direction<Dim>::all_directions()) {
      const auto logical_coords = fd::ghost_zone_logical_coordinates(
          subcell_mesh, reconstructor.ghost_zone_size(), direction);
      const auto inv_jacobian = element_map.inv_jacobian(logical_coords);
      const auto grid_coords = element_map(logical_coords);

      Variables<neighbor_tags> ghost_coords_and_inv_jacobian{
          logical_coords.get(0).size()};
      get<evolution::dg::subcell::Tags::Coordinates<Dim, Frame::Grid>>(
          ghost_coords_and_inv_jacobian) = grid_coords;
      get<evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToGrid<Dim>>(
          ghost_coords_and_inv_jacobian) = inv_jacobian;

      ghost_zone_inverse_jacobian->insert_or_assign(
          direction, ghost_coords_and_inv_jacobian);
    }
  }
};
}  // namespace evolution::dg::subcell
