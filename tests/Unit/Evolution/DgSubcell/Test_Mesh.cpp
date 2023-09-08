// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/DgSubcell/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

namespace {
template <Spectral::Basis BasisType, Spectral::Quadrature QuadratureType>
void test_mesh() {
  constexpr size_t min_num_pts =
      Spectral::minimum_number_of_points<BasisType, QuadratureType>;
  constexpr size_t max_num_pts = Spectral::maximum_number_of_points<BasisType>;
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      evolution::dg::subcell::fd::dg_mesh(
          Mesh<1>{2 * (max_num_pts - 1) - 1, Spectral::Basis::Legendre,
                  Spectral::Quadrature::CellCentered},
          BasisType, QuadratureType),
      Catch::Matchers::ContainsSubstring("The basis for computing the DG mesh "
                                         "must be FiniteDifference but got "));
  CHECK_THROWS_WITH(
      evolution::dg::subcell::fd::dg_mesh(
          Mesh<1>{2 * (max_num_pts - 1) - 1, Spectral::Basis::FiniteDifference,
                  Spectral::Quadrature::FaceCentered},
          BasisType, QuadratureType),
      Catch::Matchers::ContainsSubstring("The quadrature for computing the DG "
                                         "mesh must be CellCentered but got "));
  CHECK_THROWS_WITH(
      evolution::dg::subcell::fd::dg_mesh(
          Mesh<1>{2 * (max_num_pts - 1) - 1, Spectral::Basis::FiniteDifference,
                  Spectral::Quadrature::CellCentered},
          Spectral::Basis::FiniteDifference, QuadratureType),
      Catch::Matchers::ContainsSubstring(
          "The DG basis must be Legendre or Chebyshev but got "));
  CHECK_THROWS_WITH(
      evolution::dg::subcell::fd::dg_mesh(
          Mesh<1>{2 * (max_num_pts - 1) - 1, Spectral::Basis::FiniteDifference,
                  Spectral::Quadrature::CellCentered},
          BasisType, Spectral::Quadrature::FaceCentered),
      Catch::Matchers::ContainsSubstring(
          "The DG quadrature for computing the DG mesh must be Gauss or "
          "GaussLobatto but "));
#endif // SPECTRE_DEBUG

  for (size_t i = min_num_pts; i < max_num_pts; ++i) {
    CHECK(evolution::dg::subcell::fd::mesh(
              Mesh<1>(i, BasisType, QuadratureType)) ==
          Mesh<1>{2 * i - 1, Spectral::Basis::FiniteDifference,
                  Spectral::Quadrature::CellCentered});
    CHECK(evolution::dg::subcell::fd::mesh(
              Mesh<2>(i, BasisType, QuadratureType)) ==
          Mesh<2>{2 * i - 1, Spectral::Basis::FiniteDifference,
                  Spectral::Quadrature::CellCentered});
    CHECK(evolution::dg::subcell::fd::mesh(
              Mesh<3>(i, BasisType, QuadratureType)) ==
          Mesh<3>{2 * i - 1, Spectral::Basis::FiniteDifference,
                  Spectral::Quadrature::CellCentered});

    CHECK(evolution::dg::subcell::fd::dg_mesh(
              Mesh<1>{2 * i - 1, Spectral::Basis::FiniteDifference,
                      Spectral::Quadrature::CellCentered},
              BasisType,
              QuadratureType) == Mesh<1>(i, BasisType, QuadratureType));
    CHECK(evolution::dg::subcell::fd::dg_mesh(
              Mesh<2>{2 * i - 1, Spectral::Basis::FiniteDifference,
                      Spectral::Quadrature::CellCentered},
              BasisType,
              QuadratureType) == Mesh<2>(i, BasisType, QuadratureType));
    CHECK(evolution::dg::subcell::fd::dg_mesh(
              Mesh<3>{2 * i - 1, Spectral::Basis::FiniteDifference,
                      Spectral::Quadrature::CellCentered},
              BasisType,
              QuadratureType) == Mesh<3>(i, BasisType, QuadratureType));
  }
  CHECK(evolution::dg::subcell::fd::mesh(
            Mesh<2>({{4, 6}}, BasisType, QuadratureType)) ==
        Mesh<2>{{{7, 11}},
                Spectral::Basis::FiniteDifference,
                Spectral::Quadrature::CellCentered});
  CHECK(evolution::dg::subcell::fd::mesh(
            Mesh<3>({{4, 6, 7}}, BasisType, QuadratureType)) ==
        Mesh<3>{{{7, 11, 13}},
                Spectral::Basis::FiniteDifference,
                Spectral::Quadrature::CellCentered});

  CHECK(evolution::dg::subcell::fd::dg_mesh(
            Mesh<2>{{{7, 11}},
                    Spectral::Basis::FiniteDifference,
                    Spectral::Quadrature::CellCentered},
            BasisType,
            QuadratureType) == Mesh<2>({{4, 6}}, BasisType, QuadratureType));
  CHECK(evolution::dg::subcell::fd::dg_mesh(
            Mesh<3>{{{7, 11, 13}},
                    Spectral::Basis::FiniteDifference,
                    Spectral::Quadrature::CellCentered},
            BasisType,
            QuadratureType) == Mesh<3>({{4, 6, 7}}, BasisType, QuadratureType));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.FD.Mesh", "[Evolution][Unit]") {
  test_mesh<Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto>();
  test_mesh<Spectral::Basis::Legendre, Spectral::Quadrature::Gauss>();
  test_mesh<Spectral::Basis::Chebyshev, Spectral::Quadrature::GaussLobatto>();
  test_mesh<Spectral::Basis::Chebyshev, Spectral::Quadrature::Gauss>();
}
