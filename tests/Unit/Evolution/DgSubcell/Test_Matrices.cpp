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
#include "Evolution/DgSubcell/Matrices.hpp"
#include "Helpers/Evolution/DgSubcell/ProjectionTestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
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
      gsl::at(projection_mat, d) = std::cref(
          projection_matrix(dg_mesh.slice_through(d), subcell_mesh.extents()[d],
                            Spectral::Quadrature::CellCentered));
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
        Catch::Matchers::Contains("ghost_zone_size must be"));
    CHECK_THROWS_WITH(
        projection_matrix(Mesh<1>{3, Spectral::Basis::Legendre,
                                  Spectral::Quadrature::GaussLobatto},
                          5, 6, Side::Lower),
        Catch::Matchers::Contains("ghost_zone_size must be"));
    CHECK_THROWS_WITH(
        projection_matrix(Mesh<1>{3, Spectral::Basis::Chebyshev,
                                  Spectral::Quadrature::GaussLobatto},
                          5, 1, Side::Lower),
        Catch::Matchers::Contains(
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
            Spectral::Quadrature::FaceCentered));
      } else {
        gsl::at(projection_mat, d) = std::cref(projection_matrix(
            dg_mesh.slice_through(d), subcell_mesh.extents()[d],
            Spectral::Quadrature::CellCentered));
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
           single_recons.data(), single_recons.rows(), subcell_values.data(), 1,
           0.0, reconstructed_nodal_coeffs.data(), 1);

    CHECK_ITERABLE_CUSTOM_APPROX(expected_nodal_coeffs,
                                 reconstructed_nodal_coeffs, local_approx);
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
