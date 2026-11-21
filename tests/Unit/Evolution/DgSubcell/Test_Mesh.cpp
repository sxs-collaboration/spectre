// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <sstream>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DiscontinuousGalerkin/NumberOfSubcellPoints.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/MaximumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/MinimumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"

namespace {
using evolution::dg::auto_number_of_subcell_points_coefficients;

void print_comparison_point_computation() {
  // Prints the number of grid points for FD calculated either as:
  //   1. (2.0 / dg_min_spacing), effectively matching the grid spacing
  //   2. `evolution::dg::number_of_subcell_points()` with
  //      `subcell_points_coefficients` below, which is 2N-1 by default. 2N-1
  //       assumes the time stepper order matches the spatial order. This
  //       assumption is usually wrong about 5th or 6th order in space.
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
  // Change these to print other mappings of DG to FD grid points
  constexpr std::array<int, 3> number_of_points_coefficients =
      auto_number_of_subcell_points_coefficients;
  for (size_t i = 5; i < 16; ++i) {
    const Mesh<1> dg_mesh{i, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
    const DataVector& collocation_pts = Spectral::collocation_points(dg_mesh);
    ss << dg_mesh.extents(0) << ' '
       << 2.0 / std::abs(collocation_pts[1] - collocation_pts[0]) << " "
       << evolution::dg::number_of_subcell_points(dg_mesh.extents(0),
                                                  number_of_points_coefficients)
       << '\n';
  }
  const std::string str = ss.str();
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
  std::printf("%s\n", str.c_str());
}

template <Spectral::Basis BasisType, Spectral::Quadrature QuadratureType>
void test_mesh() {
  constexpr size_t min_num_pts =
      Spectral::minimum_number_of_points<BasisType, QuadratureType>;
  constexpr size_t max_num_pts =
      Spectral::maximum_number_of_points<BasisType, QuadratureType>;

  for (size_t i = min_num_pts; i <= max_num_pts; ++i) {
    CHECK(evolution::dg::subcell::fd::mesh(
              Mesh<1>(i, BasisType, QuadratureType),
              auto_number_of_subcell_points_coefficients) ==
          Mesh<1>{2 * i - 1, Spectral::Basis::FiniteDifference,
                  Spectral::Quadrature::CellCentered});
    CHECK(evolution::dg::subcell::fd::mesh(
              Mesh<2>(i, BasisType, QuadratureType),
              auto_number_of_subcell_points_coefficients) ==
          Mesh<2>{2 * i - 1, Spectral::Basis::FiniteDifference,
                  Spectral::Quadrature::CellCentered});
    CHECK(evolution::dg::subcell::fd::mesh(
              Mesh<3>(i, BasisType, QuadratureType),
              auto_number_of_subcell_points_coefficients) ==
          Mesh<3>{2 * i - 1, Spectral::Basis::FiniteDifference,
                  Spectral::Quadrature::CellCentered});

    // Same number of FD as DG points
    CHECK(evolution::dg::subcell::fd::mesh(
              Mesh<3>(i, BasisType, QuadratureType), {{0, 1, 0}}) ==
          Mesh<3>{i, Spectral::Basis::FiniteDifference,
                  Spectral::Quadrature::CellCentered});
  }
  for (size_t i = min_num_pts; i <= 6; ++i) {
    CHECK(evolution::dg::subcell::fd::mesh(
              Mesh<1>(i, BasisType, QuadratureType), {{1, -1, 3}}) ==
          Mesh<1>{i * i - i + 3, Spectral::Basis::FiniteDifference,
                  Spectral::Quadrature::CellCentered});
    CHECK(evolution::dg::subcell::fd::mesh(
              Mesh<2>(i, BasisType, QuadratureType), {{1, -1, 3}}) ==
          Mesh<2>{i * i - i + 3, Spectral::Basis::FiniteDifference,
                  Spectral::Quadrature::CellCentered});
    CHECK(evolution::dg::subcell::fd::mesh(
              Mesh<3>(i, BasisType, QuadratureType), {{1, -1, 3}}) ==
          Mesh<3>{i * i - i + 3, Spectral::Basis::FiniteDifference,
                  Spectral::Quadrature::CellCentered});
  }
  CHECK(evolution::dg::subcell::fd::mesh(
            Mesh<2>({{4, 6}}, BasisType, QuadratureType),
            auto_number_of_subcell_points_coefficients) ==
        Mesh<2>{{{7, 11}},
                Spectral::Basis::FiniteDifference,
                Spectral::Quadrature::CellCentered});
  CHECK(evolution::dg::subcell::fd::mesh(
            Mesh<3>({{4, 6, 7}}, BasisType, QuadratureType),
            auto_number_of_subcell_points_coefficients) ==
        Mesh<3>{{{7, 11, 13}},
                Spectral::Basis::FiniteDifference,
                Spectral::Quadrature::CellCentered});
  CHECK(evolution::dg::subcell::fd::mesh(
            Mesh<3>({{4, 6, 7}}, BasisType, QuadratureType), {{0, 3, -2}}) ==
        Mesh<3>{{{10, 16, 19}},
                Spectral::Basis::FiniteDifference,
                Spectral::Quadrature::CellCentered});

  CHECK_THROWS_WITH(evolution::dg::subcell::fd::mesh(
                        Mesh<3>(5, BasisType, QuadratureType), {{0, 1, -1}}),
                    Catch::Matchers::ContainsSubstring(
                        "must be at least the number of DG grid points (5)"));
  CHECK_THROWS_WITH(
      evolution::dg::subcell::fd::mesh(Mesh<2>(8, BasisType, QuadratureType),
                                       {{1, 0, 0}}),
      Catch::Matchers::ContainsSubstring("for 8 DG grid points must be in"));
}

template <Spectral::Basis BasisType, Spectral::Quadrature QuadratureType>
void test_cartoon_mesh() {
  for (size_t i = 2; i < 5; ++i) {
    for (const auto& [coefficients, expected_num_fd_points] : std::array{
             std::pair{auto_number_of_subcell_points_coefficients, 2 * i - 1},
             std::pair{std::array{1, 0, 1}, i * i + 1}}) {
      CAPTURE(coefficients);
      // Spherically symmetric: second and third bases are Cartoon.
      const Mesh<3> dg_spherical{
          {{i, 1, 1}},
          {BasisType, Spectral::Basis::Cartoon, Spectral::Basis::Cartoon},
          {QuadratureType, Spectral::Quadrature::SphericalSymmetry,
           Spectral::Quadrature::SphericalSymmetry}};
      const Mesh<3> subcell_spherical{
          {{expected_num_fd_points, 1, 1}},
          {Spectral::Basis::FiniteDifference, Spectral::Basis::Cartoon,
           Spectral::Basis::Cartoon},
          {Spectral::Quadrature::CellCentered,
           Spectral::Quadrature::SphericalSymmetry,
           Spectral::Quadrature::SphericalSymmetry}};
      CHECK(evolution::dg::subcell::fd::mesh(dg_spherical, coefficients) ==
            subcell_spherical);

      // Axially symmetric: only third basis is Cartoon.
      const Mesh<3> dg_axial{{{i, i, 1}},
                             {BasisType, BasisType, Spectral::Basis::Cartoon},
                             {QuadratureType, QuadratureType,
                              Spectral::Quadrature::AxialSymmetry}};
      const Mesh<3> subcell_axial{
          {{expected_num_fd_points, expected_num_fd_points, 1}},
          {Spectral::Basis::FiniteDifference, Spectral::Basis::FiniteDifference,
           Spectral::Basis::Cartoon},
          {Spectral::Quadrature::CellCentered,
           Spectral::Quadrature::CellCentered,
           Spectral::Quadrature::AxialSymmetry}};
      CHECK(evolution::dg::subcell::fd::mesh(dg_axial, coefficients) ==
            subcell_axial);
    }
  }
  // mesh(): unsupported bases return the uninitialized sentinel rather than
  // asserting, since subcell is never used on such elements.
  {
    const Mesh<3> zernike_b2_cartoon{
        {{3, 1, 1}},
        {Spectral::Basis::ZernikeB2, Spectral::Basis::Cartoon,
         Spectral::Basis::Cartoon},
        {Spectral::Quadrature::GaussRadauUpper,
         Spectral::Quadrature::SphericalSymmetry,
         Spectral::Quadrature::SphericalSymmetry}};
    CHECK(evolution::dg::subcell::fd::mesh(
              zernike_b2_cartoon, auto_number_of_subcell_points_coefficients) ==
          Mesh<3>{0_st, Spectral::Basis::Uninitialized,
                  Spectral::Quadrature::Uninitialized});
    // FiniteDifference is also not a valid DG basis, so it returns the
    // uninitialized sentinel.
    const Mesh<3> fd_cartoon{
        {{3, 3, 1}},
        {Spectral::Basis::FiniteDifference, BasisType,
         Spectral::Basis::Cartoon},
        {Spectral::Quadrature::CellCentered, QuadratureType,
         Spectral::Quadrature::AxialSymmetry}};
    CHECK(evolution::dg::subcell::fd::mesh(
              fd_cartoon, auto_number_of_subcell_points_coefficients) ==
          Mesh<3>{0_st, Spectral::Basis::Uninitialized,
                  Spectral::Quadrature::Uninitialized});
  }
}

void test_zernike_b1_cartoon_mesh() {
  for (size_t i = 2; i < 5; ++i) {
    for (const auto& [coefficients, expected_num_fd_points] : std::array{
             std::pair{auto_number_of_subcell_points_coefficients, 2 * i - 1},
             std::pair{std::array{0, 3, 1}, 3 * i + 1}}) {
      CAPTURE(coefficients);
      // Spherically symmetric: ZernikeB1 in dim 0, Cartoon in dims 1 and 2.
      const Mesh<3> dg_spherical{
          {{i, 1, 1}},
          {Spectral::Basis::ZernikeB1, Spectral::Basis::Cartoon,
           Spectral::Basis::Cartoon},
          {Spectral::Quadrature::GaussRadauUpper,
           Spectral::Quadrature::SphericalSymmetry,
           Spectral::Quadrature::SphericalSymmetry}};
      const Mesh<3> subcell_spherical{
          {{expected_num_fd_points, 1, 1}},
          {Spectral::Basis::FiniteDifference, Spectral::Basis::Cartoon,
           Spectral::Basis::Cartoon},
          {Spectral::Quadrature::CellCentered,
           Spectral::Quadrature::SphericalSymmetry,
           Spectral::Quadrature::SphericalSymmetry}};
      CHECK(evolution::dg::subcell::fd::mesh(dg_spherical, coefficients) ==
            subcell_spherical);
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.FD.Mesh", "[Evolution][Unit]") {
  test_mesh<Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto>();
  test_mesh<Spectral::Basis::Legendre, Spectral::Quadrature::Gauss>();
  test_mesh<Spectral::Basis::Chebyshev, Spectral::Quadrature::GaussLobatto>();
  test_mesh<Spectral::Basis::Chebyshev, Spectral::Quadrature::Gauss>();
  test_cartoon_mesh<Spectral::Basis::Legendre,
                    Spectral::Quadrature::GaussLobatto>();
  test_cartoon_mesh<Spectral::Basis::Legendre, Spectral::Quadrature::Gauss>();
  test_cartoon_mesh<Spectral::Basis::Chebyshev,
                    Spectral::Quadrature::GaussLobatto>();
  test_cartoon_mesh<Spectral::Basis::Chebyshev, Spectral::Quadrature::Gauss>();
  test_zernike_b1_cartoon_mesh();
  print_comparison_point_computation();

  INFO("Test dg_mesh_supports_subcell");
  {
    CHECK(evolution::dg::subcell::fd::dg_mesh_supports_subcell(Mesh<1>{
        5, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto}));
    CHECK(evolution::dg::subcell::fd::dg_mesh_supports_subcell(Mesh<2>{
        5, Spectral::Basis::Chebyshev, Spectral::Quadrature::GaussLobatto}));
    CHECK(evolution::dg::subcell::fd::dg_mesh_supports_subcell(
        Mesh<3>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss}));
    CHECK(evolution::dg::subcell::fd::dg_mesh_supports_subcell(
        Mesh<3>{{{5, 1, 1}},
                {Spectral::Basis::Legendre, Spectral::Basis::Cartoon,
                 Spectral::Basis::Cartoon},
                {Spectral::Quadrature::GaussLobatto,
                 Spectral::Quadrature::SphericalSymmetry,
                 Spectral::Quadrature::SphericalSymmetry}}));
    CHECK(evolution::dg::subcell::fd::dg_mesh_supports_subcell(
        Mesh<3>{{{5, 1, 1}},
                {Spectral::Basis::ZernikeB1, Spectral::Basis::Cartoon,
                 Spectral::Basis::Cartoon},
                {Spectral::Quadrature::GaussRadauUpper,
                 Spectral::Quadrature::SphericalSymmetry,
                 Spectral::Quadrature::SphericalSymmetry}}));

    CHECK_FALSE(evolution::dg::subcell::fd::dg_mesh_supports_subcell(Mesh<3>{
        {{5, 5, 9}},
        {Spectral::Basis::ZernikeB3, Spectral::Basis::ZernikeB3,
         Spectral::Basis::ZernikeB3},
        {Spectral::Quadrature::GaussRadauUpper, Spectral::Quadrature::Gauss,
         Spectral::Quadrature::Equiangular}}));
    CHECK_FALSE(evolution::dg::subcell::fd::dg_mesh_supports_subcell(Mesh<3>{
        {{5, 6, 11}},
        {Spectral::Basis::SphericalHarmonic, Spectral::Basis::SphericalHarmonic,
         Spectral::Basis::SphericalHarmonic},
        {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss,
         Spectral::Quadrature::Equiangular}}));
    CHECK_FALSE(evolution::dg::subcell::fd::dg_mesh_supports_subcell(
        Mesh<1>{5, Spectral::Basis::FiniteDifference,
                Spectral::Quadrature::CellCentered}));
  }

  INFO("Test fd::mesh with unsupported bases");
  {
    const Mesh<3> zernike_b3_mesh{
        {{5, 5, 9}},
        {Spectral::Basis::ZernikeB3, Spectral::Basis::ZernikeB3,
         Spectral::Basis::ZernikeB3},
        {Spectral::Quadrature::GaussRadauUpper, Spectral::Quadrature::Gauss,
         Spectral::Quadrature::Equiangular}};
    CHECK(evolution::dg::subcell::fd::mesh(
              zernike_b3_mesh, auto_number_of_subcell_points_coefficients) ==
          Mesh<3>{0_st, Spectral::Basis::Uninitialized,
                  Spectral::Quadrature::Uninitialized});

    const Mesh<3> spherical_harmonic_mesh{
        {{5, 6, 11}},
        {Spectral::Basis::SphericalHarmonic, Spectral::Basis::SphericalHarmonic,
         Spectral::Basis::SphericalHarmonic},
        {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss,
         Spectral::Quadrature::Equiangular}};
    CHECK(evolution::dg::subcell::fd::mesh(
              spherical_harmonic_mesh,
              auto_number_of_subcell_points_coefficients) ==
          Mesh<3>{0_st, Spectral::Basis::Uninitialized,
                  Spectral::Quadrature::Uninitialized});
  }
}
