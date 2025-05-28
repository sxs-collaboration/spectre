// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Side.hpp"
#include "Evolution/DgSubcell/CombineVolumeGhostData.hpp"
#include "Evolution/DgSubcell/GhostZoneLogicalCoordinates.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace {
class DummyReconstructor {
 public:
  static size_t ghost_zone_size() { return 3; }
};

template <size_t Dim>
void test(bool test_lower) {
  const auto subcell_mesh = evolution::dg::subcell::fd::mesh(::Mesh<Dim>{
      {5}, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto});
  const auto element_logical_coords = logical_coordinates(subcell_mesh);

  DataVector volume_data{Dim * element_logical_coords[0].size()};
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < element_logical_coords[0].size(); ++j) {
      volume_data[j + i * element_logical_coords[0].size()] =
          element_logical_coords[i][j];
    }
  }
  for (size_t i = 0; i < Dim; ++i) {
    const Direction<Dim> direction_to_extend{
        i, test_lower ? Side::Lower : Side::Upper};
    const auto element_ghost_coords =
        evolution::dg::subcell::fd::ghost_zone_logical_coordinates(
            subcell_mesh, DummyReconstructor::ghost_zone_size(),
            direction_to_extend);
    DataVector ghost_data{Dim * element_ghost_coords[0].size()};
    for (size_t j = 0; j < Dim; ++j) {
      for (size_t k = 0; k < element_ghost_coords[0].size(); ++k) {
        ghost_data[k + j * element_ghost_coords[0].size()] =
            element_ghost_coords[j][k];
      }
    }
    const DataVector combined_data =
        evolution::dg::subcell::combine_volume_ghost_data(
            volume_data, ghost_data, subcell_mesh.extents(),
            DummyReconstructor::ghost_zone_size(), direction_to_extend);

    auto new_extents = make_array<Dim>(subcell_mesh.extents(0));
    gsl::at(new_extents, i) =
        subcell_mesh.extents(i) + DummyReconstructor::ghost_zone_size();
    const auto extended_mesh = ::Mesh<Dim>{new_extents, subcell_mesh.basis(),
                                           subcell_mesh.quadrature()};
    auto extended_logical_coords = logical_coordinates(extended_mesh);
    const double rescale_factor =
        static_cast<double>(subcell_mesh.extents(i) +
                            DummyReconstructor::ghost_zone_size()) /
        (subcell_mesh.extents(i));
    double translation =
        static_cast<double>(DummyReconstructor::ghost_zone_size()) /
        (subcell_mesh.extents(i));
    if (test_lower) {
      translation *= -1.;
    }
    for (size_t l = 0; l < extended_logical_coords[0].size(); ++l) {
      extended_logical_coords.get(i)[l] *= rescale_factor;
      extended_logical_coords.get(i)[l] += translation;
    }

    DataVector expected_combined_data{Dim * extended_logical_coords[0].size()};
    for (size_t n = 0; n < Dim; ++n) {
      for (size_t m = 0; m < extended_logical_coords[0].size(); ++m) {
        expected_combined_data[m + n * extended_logical_coords[0].size()] =
            extended_logical_coords[n][m];
      }
    }
    CAPTURE(direction_to_extend);
    CAPTURE(test_lower);
    CHECK_ITERABLE_APPROX(combined_data, expected_combined_data);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.CombineVolumeGhostData",
                  "[Evolution][Unit]") {
  for (const bool test_lower : {false, true}) {
    test<1>(test_lower);
    test<2>(test_lower);
    test<3>(test_lower);
  }
}
