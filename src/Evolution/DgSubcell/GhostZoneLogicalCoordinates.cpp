// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/GhostZoneLogicalCoordinates.hpp"

#include <cstddef>

#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Side.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

namespace evolution::dg::subcell::fd {

template <size_t Dim>
void ghost_zone_logical_coordinates(
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::ElementLogical>*>
        ghost_logical_coords,
    const Mesh<Dim>& subcell_mesh, const size_t ghost_zone_size,
    const Direction<Dim>& direction) {
  const size_t dim_direction = direction.dimension();
  const auto subcell_extents = subcell_mesh.extents();
  const size_t subcell_extent_to_direction = subcell_extents[dim_direction];

  // Check if the `ghost_zone_size` has a valid value.
  ASSERT(ghost_zone_size <= subcell_extent_to_direction,
         " Ghost zone size ("
             << ghost_zone_size << ") is larger than the volume extent ("
             << subcell_extent_to_direction << ") to the direction");

  set_number_of_grid_points(
      ghost_logical_coords,
      ghost_zone_size *
          subcell_mesh.extents().slice_away(dim_direction).product());

  Index<Dim> ghost_zone_extents{subcell_extents};
  ghost_zone_extents[dim_direction] = ghost_zone_size;

  for (size_t d = 0; d < Dim; ++d) {
    const auto& collocation_points_in_this_dim =
        Spectral::collocation_points<Spectral::Basis::FiniteDifference,
                                     Spectral::Quadrature::CellCentered>(
            subcell_extents[d]);

    if (d == dim_direction) {
      const size_t index_offset =
          (direction.side() == Side::Upper)
              ? subcell_extent_to_direction - ghost_zone_size
              : 0;

      const double delta_x{collocation_points_in_this_dim[1] -
                           collocation_points_in_this_dim[0]};

      for (IndexIterator<Dim> index(ghost_zone_extents); index; ++index) {
        ghost_logical_coords->get(d)[index.collapsed_index()] =
            collocation_points_in_this_dim[index()[d] + index_offset] +
            direction.sign() * delta_x * ghost_zone_size;
      }
    } else {
      for (IndexIterator<Dim> index(ghost_zone_extents); index; ++index) {
        ghost_logical_coords->get(d)[index.collapsed_index()] =
            collocation_points_in_this_dim[index()[d]];
      }
    }
  }
}

template <size_t Dim>
tnsr::I<DataVector, Dim, Frame::ElementLogical> ghost_zone_logical_coordinates(
    const Mesh<Dim>& subcell_mesh, const size_t ghost_zone_size,
    const Direction<Dim>& direction) {
  tnsr::I<DataVector, Dim, Frame::ElementLogical> logical_coords(
      subcell_mesh.extents().product());
  ghost_zone_logical_coordinates(make_not_null(&logical_coords), subcell_mesh,
                                 ghost_zone_size, direction);
  return logical_coords;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                           \
  template void ghost_zone_logical_coordinates(                          \
      const gsl::not_null<                                               \
          tnsr::I<DataVector, DIM(data), Frame::ElementLogical>*>        \
          ghost_logical_coords,                                          \
      const Mesh<DIM(data)>& subcell_mesh, const size_t ghost_zone_size, \
      const Direction<DIM(data)>& direction);                            \
  template tnsr::I<DataVector, DIM(data), Frame::ElementLogical>         \
  ghost_zone_logical_coordinates(const Mesh<DIM(data)>& subcell_mesh,    \
                                 const size_t ghost_zone_size,           \
                                 const Direction<DIM(data)>& direction);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef DIM
#undef INSTANTIATION
}  // namespace evolution::dg::subcell::fd
