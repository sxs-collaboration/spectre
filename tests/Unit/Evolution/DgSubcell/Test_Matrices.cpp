// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Domain/LogicalCoordinates.hpp"
#include "Evolution/DgSubcell/Matrices.hpp"
#include "Helpers/Evolution/DgSubcell/ProjectionTestHelpers.hpp"
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
    const Mesh<Dim> dg_mesh{num_pts_1d, BasisType, QuadratureType};
    const auto logical_coords = logical_coordinates(dg_mesh);
    const size_t num_subcells_1d = 2 * num_pts_1d - 1;
    const Mesh<Dim> subcell_mesh(num_subcells_1d,
                                 Spectral::Basis::FiniteDifference,
                                 Spectral::Quadrature::CellCentered);
    const size_t num_subcells = subcell_mesh.number_of_grid_points();
    const DataVector nodal_coeffs =
        TestHelpers::evolution::dg::subcell::cell_values(dg_mesh.extents(0) - 2,
                                                         logical_coords);

    const Matrix& proj_matrix =
        projection_matrix(dg_mesh, subcell_mesh.extents());

    DataVector cell_centered_values(num_subcells, 0.0);
    dgemv_('N', proj_matrix.rows(), proj_matrix.columns(), 1.0,
           proj_matrix.data(), proj_matrix.rows(), nodal_coeffs.data(), 1, 0.0,
           cell_centered_values.data(), 1);

    CHECK_ITERABLE_APPROX(
        cell_centered_values,
        TestHelpers::evolution::dg::subcell::cell_values(
            dg_mesh.extents(0) - 2, logical_coordinates(subcell_mesh)));
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
