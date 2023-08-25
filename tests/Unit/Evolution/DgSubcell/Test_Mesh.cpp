// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/DgSubcell/Mesh.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

namespace {
template <SpatialDiscretization::Basis BasisType,
          SpatialDiscretization::Quadrature QuadratureType>
void test_mesh() {
  constexpr size_t min_num_pts =
      Spectral::minimum_number_of_points<BasisType, QuadratureType>;
  constexpr size_t max_num_pts = Spectral::maximum_number_of_points<BasisType>;
  for (size_t i = min_num_pts; i < max_num_pts; ++i) {
    CHECK(evolution::dg::subcell::fd::mesh(
              Mesh<1>(i, BasisType, QuadratureType)) ==
          Mesh<1>{2 * i - 1, SpatialDiscretization::Basis::FiniteDifference,
                  SpatialDiscretization::Quadrature::CellCentered});
    CHECK(evolution::dg::subcell::fd::mesh(
              Mesh<2>(i, BasisType, QuadratureType)) ==
          Mesh<2>{2 * i - 1, SpatialDiscretization::Basis::FiniteDifference,
                  SpatialDiscretization::Quadrature::CellCentered});
    CHECK(evolution::dg::subcell::fd::mesh(
              Mesh<3>(i, BasisType, QuadratureType)) ==
          Mesh<3>{2 * i - 1, SpatialDiscretization::Basis::FiniteDifference,
                  SpatialDiscretization::Quadrature::CellCentered});
  }
  CHECK(evolution::dg::subcell::fd::mesh(
            Mesh<2>({{4, 6}}, BasisType, QuadratureType)) ==
        Mesh<2>{{{7, 11}},
                SpatialDiscretization::Basis::FiniteDifference,
                SpatialDiscretization::Quadrature::CellCentered});
  CHECK(evolution::dg::subcell::fd::mesh(
            Mesh<3>({{4, 6, 7}}, BasisType, QuadratureType)) ==
        Mesh<3>{{{7, 11, 13}},
                SpatialDiscretization::Basis::FiniteDifference,
                SpatialDiscretization::Quadrature::CellCentered});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.FD.Mesh", "[Evolution][Unit]") {
  test_mesh<SpatialDiscretization::Basis::Legendre,
            SpatialDiscretization::Quadrature::GaussLobatto>();
  test_mesh<SpatialDiscretization::Basis::Legendre,
            SpatialDiscretization::Quadrature::Gauss>();
  test_mesh<SpatialDiscretization::Basis::Chebyshev,
            SpatialDiscretization::Quadrature::GaussLobatto>();
  test_mesh<SpatialDiscretization::Basis::Chebyshev,
            SpatialDiscretization::Quadrature::Gauss>();
}
