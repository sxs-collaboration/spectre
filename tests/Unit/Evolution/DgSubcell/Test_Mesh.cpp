// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <sstream>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/MaximumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/MinimumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"

namespace {
void print_comparison_point_computation() {
  // Prints the number of grid points for FD calculated either as:
  //   1. (2.0 / dg_min_spacing), effectively matching the grid spacing
  //   2. (2 * num_dg_points - 1), which assumes the time stepper order
  //      matches the spatial order. This assumption is usually wrong about
  //      5th or 6th order in space.
  //
  // DG points | (2.0 / dg_min_spacing) | (2 * num_dg_points - 1)
  //  5        |       5.79129          |     9
  //  6        |       8.51264          |     11
  //  7        |       11.7802          |     13
  //  8        |       15.5933          |     15
  //  9        |       19.9517          |     17
  //  10       |       24.8552          |     19
  //  11       |       30.3037          |     21
  //  12       |       36.2972          |     23
  //  13       |       42.8356          |     25
  //  14       |       49.9189          |     27
  //  15       |       57.5472          |     29
  //
  // Clearly at high-order DG we could have way more FD grid points to match
  // the spatial grid spacing. It's not clear to Nils D. whether the right
  // thing to do is to always match the DG grid spacing, or transition from
  // one method to the other at some point.
  std::stringstream ss;
  for (size_t i = 5; i < 16; ++i) {
    const Mesh<1> dg_mesh{i, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
    const DataVector& collocation_pts = Spectral::collocation_points(dg_mesh);
    ss << dg_mesh.extents(0) << ' '
       << 2.0 / std::abs(collocation_pts[1] - collocation_pts[0]) << " "
       << (2 * dg_mesh.extents(0) - 1) << '\n';
  }
  const std::string str = ss.str();
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
  std::printf("%s\n", str.c_str());
}

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
  print_comparison_point_computation();
}
