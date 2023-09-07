// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/DgSubcell/GhostZoneLogicalCoordinates.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/SliceTensor.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"

namespace evolution::dg::subcell::fd {
namespace {
template <size_t Dim>
void test() {
  // Create a subcell mesh
  const Mesh<Dim> dg_mesh{3, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> subcell_mesh = mesh(dg_mesh);
  const Index<Dim> subcell_extents = subcell_mesh.extents();

  for (const auto& direction : Direction<Dim>::all_directions()) {
    const size_t dim_direction = direction.dimension();

    // iterate ghost zone size up to the maximum possible value (mesh extent to
    // the direction)
    for (size_t ghost_zone_size = 1;
         ghost_zone_size <= subcell_extents[dim_direction]; ++ghost_zone_size) {
      const tnsr::I<DataVector, Dim, Frame::ElementLogical> ghost_zone_coords =
          ghost_zone_logical_coordinates(subcell_mesh, ghost_zone_size,
                                         direction);

      // grid extents of ghost zone
      Index<Dim> ghost_zone_extents{subcell_extents};
      ghost_zone_extents[dim_direction] = ghost_zone_size;

      // Check the computed ghost zone coords slice-by-slice.
      // First we get the outermost slice of subcell (volume) logical
      // coordinates to copy the coordinate components that remain same. i.e.
      // For 3D if `direction` is along x-axis, y and z components would remain
      // same as volume slice. (*)
      auto expected_coordinates = slice_tensor_for_subcell(
          logical_coordinates(subcell_mesh), subcell_extents, 1, direction);

      for (size_t i_slice = 0; i_slice < ghost_zone_size; ++i_slice) {
        auto ghost_zone_coords_ith_slice = data_on_slice(
            ghost_zone_coords, ghost_zone_extents, dim_direction, i_slice);

        // (*) Here we tweak the components that need to be shifted.
        // - Since subcell mesh is cell-centered, it has logical coordinate
        //   values [-0.8, -0.4, 0, 0.4, 0.8]. Depending on the `direction`,
        //   we choose either -0.8 or 0.8 as a fiducial point.
        // - Grid spacing for subcell mesh is 0.4
        expected_coordinates.get(dim_direction) =
            direction.side() == Side::Upper
                ? 0.8 + 0.4 * (i_slice + 1)
                : -0.8 - 0.4 * (ghost_zone_size - i_slice);

        CHECK_ITERABLE_APPROX(expected_coordinates,
                              ghost_zone_coords_ith_slice);
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.Fd.GhostLogicalCoords",
                  "[Evolution][Unit]") {
  test<1>();
  test<2>();
  test<3>();
}

}  // namespace
}  // namespace evolution::dg::subcell::fd
