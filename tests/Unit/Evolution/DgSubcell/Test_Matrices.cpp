// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Side.hpp"
#include "Evolution/DgSubcell/Matrices.hpp"
#include "Helpers/Evolution/DgSubcell/ProjectionTestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionValue.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/MinimumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Parity.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeArray.hpp"

namespace evolution::dg::subcell::fd {
namespace {
template <size_t MaxPts, size_t Dim, Spectral::Basis BasisType,
          Spectral::Quadrature QuadratureType>
void test_projection_matrix() {
  CAPTURE(Dim);
  CAPTURE(BasisType);
  CAPTURE(QuadratureType);

  for (size_t num_pts_1d = std::max(
           static_cast<size_t>(2),
           Spectral::minimum_number_of_points<BasisType, QuadratureType>);
       num_pts_1d < MaxPts + 1; ++num_pts_1d) {
    CAPTURE(num_pts_1d);
    const Mesh<Dim> dg_mesh{num_pts_1d, BasisType, QuadratureType};
    const auto logical_coords = logical_coordinates(dg_mesh);
    const size_t num_subcells_1d = 2 * num_pts_1d - 1;
    CAPTURE(num_subcells_1d);
    const Mesh<Dim> subcell_mesh(num_subcells_1d,
                                 Spectral::Basis::FiniteDifference,
                                 Spectral::Quadrature::CellCentered);
    const size_t num_subcells = subcell_mesh.number_of_grid_points();
    const DataVector nodal_coeffs =
        TestHelpers::evolution::dg::subcell::cell_values(dg_mesh.extents(0) - 2,
                                                         logical_coords);

    Matrix empty{};
    auto projection_mat = make_array<Dim>(std::cref(empty));
    for (size_t d = 0; d < Dim; ++d) {
      gsl::at(projection_mat, d) = std::cref(projection_matrix(
          dg_mesh.slice_through(d), subcell_mesh.extents()[d],
          Spectral::Quadrature::CellCentered, Spectral::Parity::Uninitialized));
    }
    DataVector cell_centered_values(num_subcells, 0.0);
    apply_matrices(make_not_null(&cell_centered_values), projection_mat,
                   nodal_coeffs, dg_mesh.extents());

    const DataVector expected_values =
        TestHelpers::evolution::dg::subcell::cell_values(
            dg_mesh.extents(0) - 2, logical_coordinates(subcell_mesh));
    CHECK_ITERABLE_APPROX(cell_centered_values, expected_values);

    if constexpr (Dim == 1) {
      // Check projecting ghost cells. Only do in 1d since the test becomes
      // rather error-prone and tedious in higher dimensions, and the operation
      // is dim-by-dim handled by apply_matrices.
      for (size_t ghost_points = 2;
           num_subcells_1d > 4 and
           ghost_points <= std::min(5_st, num_subcells_1d - 2);
           ++ghost_points) {
        CAPTURE(ghost_points);
        for (const Side side : {Side::Lower, Side::Upper}) {
          CAPTURE(side);
          CAPTURE(expected_values);
          DataVector expected_ghost_values(ghost_points);
          for (size_t i = 0; i < ghost_points; ++i) {
            expected_ghost_values[i] =
                expected_values[side == Side::Lower
                                    ? i
                                    : (num_subcells_1d - ghost_points + i)];
          }
          DataVector ghost_cell_centered_values(ghost_points, 0.0);
          auto ghost_projection_mat = make_array<Dim>(std::cref(empty));
          ghost_projection_mat[0] = std::cref(projection_matrix(
              dg_mesh, subcell_mesh.extents(0), ghost_points, side));
          apply_matrices(make_not_null(&ghost_cell_centered_values),
                         ghost_projection_mat, nodal_coeffs, dg_mesh.extents());
          CHECK_ITERABLE_APPROX(ghost_cell_centered_values,
                                expected_ghost_values);
        }
      }
    }
  }
#ifdef SPECTRE_DEBUG
  if constexpr (Dim == 1) {
    CHECK_THROWS_WITH(
        projection_matrix(Mesh<1>{3, Spectral::Basis::Legendre,
                                  Spectral::Quadrature::GaussLobatto},
                          5, 1, Side::Lower),
        Catch::Matchers::ContainsSubstring("ghost_zone_size must be"));
    CHECK_THROWS_WITH(
        projection_matrix(Mesh<1>{3, Spectral::Basis::Legendre,
                                  Spectral::Quadrature::GaussLobatto},
                          5, 6, Side::Lower),
        Catch::Matchers::ContainsSubstring("ghost_zone_size must be"));
    CHECK_THROWS_WITH(
        projection_matrix(Mesh<1>{3, Spectral::Basis::Chebyshev,
                                  Spectral::Quadrature::GaussLobatto},
                          5, 1, Side::Lower),
        Catch::Matchers::ContainsSubstring(
            "FD Subcell projection only supports Legendre basis"));
  }
#endif
}

template <size_t MaxPts, size_t Dim, size_t Face_Dim, Spectral::Basis BasisType,
          Spectral::Quadrature QuadratureType>
void test_projection_matrix_to_face() {
  CAPTURE(Dim);
  CAPTURE(BasisType);
  CAPTURE(QuadratureType);

  for (size_t num_pts_1d = std::max(
           static_cast<size_t>(2),
           Spectral::minimum_number_of_points<BasisType, QuadratureType>);
       num_pts_1d < MaxPts + 1; ++num_pts_1d) {
    CAPTURE(num_pts_1d);
    const Mesh<Dim> dg_mesh{num_pts_1d, BasisType, QuadratureType};
    const auto logical_coords = logical_coordinates(dg_mesh);
    const size_t num_subcells_1d_face = 2 * num_pts_1d;
    const size_t num_subcells_1d_cell = 2 * num_pts_1d - 1;
    CAPTURE(num_subcells_1d_face);
    CAPTURE(num_subcells_1d_cell);

    std::array<size_t, Dim> extents{};
    std::array<Spectral::Basis, Dim> basis{};
    std::array<Spectral::Quadrature, Dim> quadrature{};
    for (size_t d = 0; d < Dim; d++) {
      basis[d] = Spectral::Basis::FiniteDifference;
      if (d == Face_Dim) {
        extents[d] = num_subcells_1d_face;
        quadrature[d] = Spectral::Quadrature::FaceCentered;
      } else {
        extents[d] = num_subcells_1d_cell;
        quadrature[d] = Spectral::Quadrature::CellCentered;
      }
    }

    const Mesh<Dim> subcell_mesh(extents, basis, quadrature);
    const size_t num_subcells = subcell_mesh.number_of_grid_points();
    const DataVector nodal_coeffs =
        TestHelpers::evolution::dg::subcell::cell_values(dg_mesh.extents(0) - 2,
                                                         logical_coords);

    Matrix empty{};
    auto projection_mat = make_array<Dim>(std::cref(empty));
    for (size_t d = 0; d < Dim; ++d) {
      if (d == Face_Dim) {
        gsl::at(projection_mat, d) = std::cref(projection_matrix(
            dg_mesh.slice_through(d), subcell_mesh.extents()[d],
            Spectral::Quadrature::FaceCentered,
            Spectral::Parity::Uninitialized));
      } else {
        gsl::at(projection_mat, d) = std::cref(projection_matrix(
            dg_mesh.slice_through(d), subcell_mesh.extents()[d],
            Spectral::Quadrature::CellCentered,
            Spectral::Parity::Uninitialized));
      }
    }
    DataVector subcell_values(num_subcells, 0.0);
    apply_matrices(make_not_null(&subcell_values), projection_mat, nodal_coeffs,
                   dg_mesh.extents());

    const DataVector expected_values =
        TestHelpers::evolution::dg::subcell::cell_values(
            dg_mesh.extents(0) - 2, logical_coordinates(subcell_mesh));
    CHECK_ITERABLE_APPROX(subcell_values, expected_values);
  }
}

template <size_t MaxPts, size_t Dim, Spectral::Basis BasisType,
          Spectral::Quadrature QuadratureType>
void reconstruction_matrix(const double eps) {
  CAPTURE(Dim);
  CAPTURE(BasisType);
  CAPTURE(QuadratureType);
  Approx local_approx = Approx::custom().epsilon(eps).scale(1.);

  for (size_t num_pts_1d = std::max(
           static_cast<size_t>(2),
           Spectral::minimum_number_of_points<BasisType, QuadratureType>);
       num_pts_1d < MaxPts + 1; ++num_pts_1d) {
    CAPTURE(MaxPts);
    CAPTURE(num_pts_1d);
    const Mesh<Dim> dg_mesh{num_pts_1d, BasisType, QuadratureType};
    const size_t num_pts = dg_mesh.number_of_grid_points();
    const auto logical_coords = logical_coordinates(dg_mesh);
    const size_t num_subcells_1d = 2 * num_pts_1d - 1;
    const Mesh<Dim> subcell_mesh(num_subcells_1d,
                                 Spectral::Basis::FiniteDifference,
                                 Spectral::Quadrature::CellCentered);
    // Our FD reconstruction scheme can integrate polynomials up to degree 6
    // exactly. However, we want to verify that if we have more than 8 grid
    // points on the DG grid that we still are able to recover the correct
    // solution.
    const DataVector expected_nodal_coeffs =
        TestHelpers::evolution::dg::subcell::cell_values(
            std::min(dg_mesh.extents(0) - 2, 6_st), logical_coords);
    const DataVector subcell_values =
        TestHelpers::evolution::dg::subcell::cell_values(
            std::min(dg_mesh.extents(0) - 2, 6_st),
            logical_coordinates(subcell_mesh));

    const Matrix& single_recons =
        subcell::fd::reconstruction_matrix(dg_mesh, subcell_mesh.extents());

    DataVector reconstructed_nodal_coeffs(num_pts);
    dgemv_('N', single_recons.rows(), single_recons.columns(), 1.0,
           single_recons.data(), single_recons.spacing(), subcell_values.data(),
           1, 0.0, reconstructed_nodal_coeffs.data(), 1);

    CHECK_ITERABLE_CUSTOM_APPROX(expected_nodal_coeffs,
                                 reconstructed_nodal_coeffs, local_approx);
  }
}

void test_cartoon_matrices() {
  for (const auto quad : {Spectral::Quadrature::SphericalSymmetry,
                          Spectral::Quadrature::AxialSymmetry}) {
    // projection_matrix for a Cartoon dimension returns a 1x1 identity matrix.
    const Mesh<1> cartoon_mesh{1, Spectral::Basis::Cartoon, quad};
    const Matrix& proj_mat =
        projection_matrix(cartoon_mesh, /*subcell_extents=*/1, quad,
                          Spectral::Parity::Uninitialized);
    REQUIRE(proj_mat.rows() == 1);
    REQUIRE(proj_mat.columns() == 1);
    CHECK(proj_mat(0, 0) == approx(1.0));
    // reconstruction_matrix for a Cartoon 1D mesh returns a 1x1 identity
    // matrix.
    const Index<1> cartoon_subcell_extents{1};
    const Matrix& rec_mat = subcell::fd::reconstruction_matrix(
        cartoon_mesh, cartoon_subcell_extents);
    REQUIRE(rec_mat.rows() == 1);
    REQUIRE(rec_mat.columns() == 1);
    CHECK(rec_mat(0, 0) == approx(1.0));
  }
#ifdef SPECTRE_DEBUG
  // projection_matrix: unsupported subcell_quadrature.
  CHECK_THROWS_WITH(
      projection_matrix(Mesh<1>{3, Spectral::Basis::Legendre,
                                Spectral::Quadrature::GaussLobatto},
                        5, Spectral::Quadrature::GaussLobatto,
                        Spectral::Parity::Uninitialized),
      Catch::Matchers::ContainsSubstring(
          "subcell_quadrature option in projection_matrix should be"));
  // projection_matrix: unsupported basis (not Legendre, Cartoon, or ZernikeB1).
  CHECK_THROWS_WITH(
      projection_matrix(Mesh<1>{3, Spectral::Basis::Chebyshev,
                                Spectral::Quadrature::GaussLobatto},
                        5, Spectral::Quadrature::CellCentered,
                        Spectral::Parity::Uninitialized),
      Catch::Matchers::ContainsSubstring(
          "FD Subcell projection only supports Legendre, Cartoon, or "
          "ZernikeB1"));
  // projection_matrix: ZernikeB1 requires non-Uninitialized parity.
  CHECK_THROWS_WITH(
      projection_matrix(Mesh<1>{3, Spectral::Basis::ZernikeB1,
                                Spectral::Quadrature::GaussRadauUpper},
                        5, Spectral::Quadrature::CellCentered,
                        Spectral::Parity::Uninitialized),
      Catch::Matchers::ContainsSubstring(
          "Parity must be set when using ZernikeB1"));
  // reconstruction_matrix: unsupported basis at dim 0.
  CHECK_THROWS_WITH(
      subcell::fd::reconstruction_matrix(
          Mesh<1>{3, Spectral::Basis::Chebyshev,
                  Spectral::Quadrature::GaussLobatto},
          Index<1>{5}),
      Catch::Matchers::ContainsSubstring(
          "FD Subcell reconstruction only supports Legendre or Cartoon bases"));
  // reconstruction_matrix 1D: non-uniform subcell extents (trivially can't
  // fail for 1D since Index<1> is always "uniform", so test non-uniform mesh
  // instead via 2D).
  CHECK_THROWS_WITH(
      subcell::fd::reconstruction_matrix(
          Mesh<2>{{3, 4},
                  {Spectral::Basis::Legendre, Spectral::Basis::Legendre},
                  {Spectral::Quadrature::GaussLobatto,
                   Spectral::Quadrature::GaussLobatto}},
          Index<2>{5}),
      Catch::Matchers::ContainsSubstring(
          "The subcell mesh must have isotropic basis"));
  CHECK_THROWS_WITH(
      subcell::fd::reconstruction_matrix(
          Mesh<2>{3, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto},
          Index<2>{{5, 7}}),
      Catch::Matchers::ContainsSubstring("The subcell mesh must be uniform"));
#endif
}

// Tests projection for 3D meshes with trailing Cartoon dimensions:
// {Legendre, Legendre, Cartoon} (axial symmetry) and
// {Legendre, Cartoon, Cartoon} (spherical symmetry).
// Reconstruction is not tested here because DimByDim reconstruction passes
// 1D slices, which are already covered by test_cartoon_matrices.
template <size_t MaxPts, Spectral::Quadrature QuadratureType>
void test_cartoon_mixed_matrices() {
  for (size_t num_pts_1d = std::max(
           static_cast<size_t>(2),
           Spectral::minimum_number_of_points<Spectral::Basis::Legendre,
                                              QuadratureType>);
       num_pts_1d < MaxPts + 1; ++num_pts_1d) {
    CAPTURE(num_pts_1d);
    const size_t num_subcells_1d = 2 * num_pts_1d - 1;

    for (const bool axial : {true, false}) {
      CAPTURE(axial);
      // axial  => {Legendre, Legendre, Cartoon}, AxialSymmetry
      // !axial => {Legendre, Cartoon,  Cartoon}, SphericalSymmetry
      const std::array<size_t, 3> dg_extents_arr{
          num_pts_1d, axial ? num_pts_1d : 1_st, 1_st};
      const std::array<Spectral::Basis, 3> bases{
          Spectral::Basis::Legendre,
          axial ? Spectral::Basis::Legendre : Spectral::Basis::Cartoon,
          Spectral::Basis::Cartoon};
      const auto cartoon_quad = axial ? Spectral::Quadrature::AxialSymmetry
                                      : Spectral::Quadrature::SphericalSymmetry;
      const std::array<Spectral::Quadrature, 3> quadratures{
          QuadratureType, axial ? QuadratureType : cartoon_quad, cartoon_quad};

      const Mesh<3> dg_mesh{dg_extents_arr, bases, quadratures};
      const auto logical_coords = logical_coordinates(dg_mesh);

      const std::array<size_t, 3> subcell_extents_arr{
          num_subcells_1d, axial ? num_subcells_1d : 1_st, 1_st};
      const Index<3> subcell_extents{subcell_extents_arr};
      const size_t num_subcells = subcell_extents.product();

      const size_t poly_degree = std::min(num_pts_1d - 2, 6_st);
      const DataVector nodal_coeffs =
          TestHelpers::evolution::dg::subcell::cell_values(poly_degree,
                                                           logical_coords);

      // Build projection matrices per dimension.
      const Matrix empty{};
      auto projection_mat = make_array<3>(std::cref(empty));
      for (size_t d = 0; d < 3; ++d) {
        const auto subcell_quad = gsl::at(bases, d) == Spectral::Basis::Cartoon
                                      ? gsl::at(quadratures, d)
                                      : Spectral::Quadrature::CellCentered;
        gsl::at(projection_mat, d) = std::cref(
            projection_matrix(dg_mesh.slice_through(d), subcell_extents[d],
                              subcell_quad, Spectral::Parity::Uninitialized));
      }

      // Project DG -> subcell.
      DataVector subcell_values(num_subcells, 0.0);
      apply_matrices(make_not_null(&subcell_values), projection_mat,
                     nodal_coeffs, dg_mesh.extents());

      // The projected values should match the test polynomial evaluated at the
      // subcell collocation points.
      const std::array<Spectral::Basis, 3> subcell_bases{
          Spectral::Basis::FiniteDifference,
          axial ? Spectral::Basis::FiniteDifference : Spectral::Basis::Cartoon,
          Spectral::Basis::Cartoon};
      const std::array<Spectral::Quadrature, 3> subcell_quads{
          Spectral::Quadrature::CellCentered,
          axial ? Spectral::Quadrature::CellCentered : cartoon_quad,
          cartoon_quad};
      const Mesh<3> subcell_mesh{subcell_extents_arr, subcell_bases,
                                 subcell_quads};
      const DataVector expected_subcell_values =
          TestHelpers::evolution::dg::subcell::cell_values(
              poly_degree, logical_coordinates(subcell_mesh));
      CHECK_ITERABLE_APPROX(subcell_values, expected_subcell_values);
    }
  }
#ifdef SPECTRE_DEBUG
  // reconstruction_matrix 3D spherical: cartoon subcell extents must be 1.
  CHECK_THROWS_WITH(
      subcell::fd::reconstruction_matrix(
          Mesh<3>{{3, 1, 1},
                  {Spectral::Basis::Legendre, Spectral::Basis::Cartoon,
                   Spectral::Basis::Cartoon},
                  {QuadratureType, Spectral::Quadrature::SphericalSymmetry,
                   Spectral::Quadrature::SphericalSymmetry}},
          Index<3>{{5, 3, 1}}),
      Catch::Matchers::ContainsSubstring(
          "The subcell extents are neither isotropic nor a valid cartoon "
          "pattern"));
  // reconstruction_matrix 3D axial: non-uniform DG extents in non-cartoon dims.
  CHECK_THROWS_WITH(
      subcell::fd::reconstruction_matrix(
          Mesh<3>{{3, 4, 1},
                  {Spectral::Basis::Legendre, Spectral::Basis::Legendre,
                   Spectral::Basis::Cartoon},
                  {QuadratureType, QuadratureType,
                   Spectral::Quadrature::AxialSymmetry}},
          Index<3>{{5, 5, 1}}),
      Catch::Matchers::ContainsSubstring(
          "The non-cartoon subcell sub-mesh must have isotropic basis"));
  // reconstruction_matrix 3D axial: subcell extents not {n,n,1}.
  CHECK_THROWS_WITH(
      subcell::fd::reconstruction_matrix(
          Mesh<3>{{3, 3, 1},
                  {Spectral::Basis::Legendre, Spectral::Basis::Legendre,
                   Spectral::Basis::Cartoon},
                  {QuadratureType, QuadratureType,
                   Spectral::Quadrature::AxialSymmetry}},
          Index<3>{{5, 7, 1}}),
      Catch::Matchers::ContainsSubstring(
          "The subcell extents are neither isotropic nor a valid cartoon"));
  // reconstruction_matrix 3D axial: cartoon subcell extent not 1.
  CHECK_THROWS_WITH(
      subcell::fd::reconstruction_matrix(
          Mesh<3>{{3, 3, 1},
                  {Spectral::Basis::Legendre, Spectral::Basis::Legendre,
                   Spectral::Basis::Cartoon},
                  {QuadratureType, QuadratureType,
                   Spectral::Quadrature::AxialSymmetry}},
          Index<3>{{5, 5, 3}}),
      Catch::Matchers::ContainsSubstring(
          "The subcell mesh must be uniform but is"));
#endif
}

// Tests projection from a ZernikeB1/GaussRadauUpper DG mesh to a FD subcell
// mesh for both Even (m=0) and Odd (m=1) parities.
void test_zernike_b1_projection_matrix() {
  constexpr Spectral::Basis basis = Spectral::Basis::ZernikeB1;
  constexpr Spectral::Quadrature quadrature =
      Spectral::Quadrature::GaussRadauUpper;
  const Approx custom_approx = Approx::custom().epsilon(1.0e-11).scale(1.);

  for (size_t num_pts_1d =
           Spectral::minimum_number_of_points<basis, quadrature>;
       num_pts_1d <= 6; ++num_pts_1d) {
    CAPTURE(num_pts_1d);
    const Mesh<1> dg_mesh{num_pts_1d, basis, quadrature};
    const auto xi_dg = get<0>(logical_coordinates(dg_mesh));
    const size_t num_subcells_1d = 2 * num_pts_1d - 1;
    const Mesh<1> subcell_mesh{num_subcells_1d,
                               Spectral::Basis::FiniteDifference,
                               Spectral::Quadrature::CellCentered};
    const auto xi_fd = get<0>(logical_coordinates(subcell_mesh));

    const Matrix empty{};
    auto even_projection_mat = make_array<1>(std::cref(empty));
    even_projection_mat[0] = std::cref(projection_matrix(
        dg_mesh, subcell_mesh.extents()[0], Spectral::Quadrature::CellCentered,
        Spectral::Parity::Even));
    auto odd_projection_mat = make_array<1>(std::cref(empty));
    odd_projection_mat[0] = std::cref(projection_matrix(
        dg_mesh, subcell_mesh.extents()[0], Spectral::Quadrature::CellCentered,
        Spectral::Parity::Odd));

    for (size_t k = 0; k < num_pts_1d; ++k) {
      CAPTURE(k);
      // Even parity (m=0): basis functions Q^0_{2k}
      const DataVector f_even_dg =
          Spectral::compute_basis_function_value<basis>(2 * k, 0_st, xi_dg);
      const DataVector f_even_fd_expected =
          Spectral::compute_basis_function_value<basis>(2 * k, 0_st, xi_fd);
      DataVector f_even_fd(num_subcells_1d, 0.0);
      apply_matrices(make_not_null(&f_even_fd), even_projection_mat, f_even_dg,
                     dg_mesh.extents());
      CHECK_ITERABLE_CUSTOM_APPROX(f_even_fd, f_even_fd_expected,
                                   custom_approx);

      // Odd parity (m=1): basis functions Q^1_{2k+1}
      const DataVector f_odd_dg =
          Spectral::compute_basis_function_value<basis>(2 * k + 1, 1_st, xi_dg);
      const DataVector f_odd_fd_expected =
          Spectral::compute_basis_function_value<basis>(2 * k + 1, 1_st, xi_fd);
      DataVector f_odd_fd(num_subcells_1d, 0.0);
      apply_matrices(make_not_null(&f_odd_fd), odd_projection_mat, f_odd_dg,
                     dg_mesh.extents());
      CHECK_ITERABLE_CUSTOM_APPROX(f_odd_fd, f_odd_fd_expected, custom_approx);
    }
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.Fd.ProjectionMatrix",
                  "[Evolution][Unit]") {
  test_projection_matrix<10, 1, Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto>();
  test_projection_matrix<10, 1, Spectral::Basis::Legendre,
                         Spectral::Quadrature::Gauss>();

  test_projection_matrix<10, 2, Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto>();
  test_projection_matrix<10, 2, Spectral::Basis::Legendre,
                         Spectral::Quadrature::Gauss>();

  test_projection_matrix<5, 3, Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto>();
  test_projection_matrix<5, 3, Spectral::Basis::Legendre,
                         Spectral::Quadrature::Gauss>();
  test_projection_matrix_to_face<10, 1, 0, Spectral::Basis::Legendre,
                                 Spectral::Quadrature::GaussLobatto>();
  test_projection_matrix_to_face<10, 1, 0, Spectral::Basis::Legendre,
                                 Spectral::Quadrature::Gauss>();
  test_projection_matrix_to_face<5, 3, 0, Spectral::Basis::Legendre,
                                 Spectral::Quadrature::GaussLobatto>();
  test_projection_matrix_to_face<5, 3, 0, Spectral::Basis::Legendre,
                                 Spectral::Quadrature::Gauss>();
  test_projection_matrix_to_face<5, 3, 1, Spectral::Basis::Legendre,
                                 Spectral::Quadrature::GaussLobatto>();
  test_projection_matrix_to_face<5, 3, 1, Spectral::Basis::Legendre,
                                 Spectral::Quadrature::Gauss>();
  test_projection_matrix_to_face<5, 3, 2, Spectral::Basis::Legendre,
                                 Spectral::Quadrature::GaussLobatto>();
  test_projection_matrix_to_face<5, 3, 2, Spectral::Basis::Legendre,
                                 Spectral::Quadrature::Gauss>();
  test_cartoon_matrices();
  test_cartoon_mixed_matrices<10, Spectral::Quadrature::GaussLobatto>();
  test_cartoon_mixed_matrices<10, Spectral::Quadrature::Gauss>();
  test_zernike_b1_projection_matrix();
}

// [[TimeOut, 10]]
SPECTRE_TEST_CASE("Unit.Evolution.Subcell.Fd.ReconstructionMatrix",
                  "[Evolution][Unit]") {
  // Timeout is increased slightly so we can test the 3d 5 points per dim case.
  // Normally the test completes in less than 2 seconds on debug builds.
  // However, if ASAN is on, this time roughly doubles and we want to avoid
  // timeouts there.
  reconstruction_matrix<10, 1, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto>(1.0e-13);
  reconstruction_matrix<10, 1, Spectral::Basis::Legendre,
                        Spectral::Quadrature::Gauss>(1.0e-13);

  reconstruction_matrix<10, 2, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto>(1.0e-10);
  reconstruction_matrix<10, 2, Spectral::Basis::Legendre,
                        Spectral::Quadrature::Gauss>(1.0e-10);

  reconstruction_matrix<5, 3, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto>(1.0e-11);
  reconstruction_matrix<4, 3, Spectral::Basis::Legendre,
                        Spectral::Quadrature::Gauss>(1.0e-11);
}
}  // namespace
}  // namespace evolution::dg::subcell::fd
