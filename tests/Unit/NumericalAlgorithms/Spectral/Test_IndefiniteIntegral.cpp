// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Blas.hpp"

namespace Spectral {
namespace {
DataVector integrate(const DataVector& u, const Mesh<1>& mesh) noexcept {
  const size_t num_pts = mesh.number_of_grid_points();
  DataVector result(num_pts, 0.0);
  const Matrix& indef_int_with_constant = integration_matrix(mesh);
  dgemv_('N', num_pts, num_pts, 1.0, indef_int_with_constant.data(), num_pts,
         u.data(), 1, 0.0, result.data(), 1);
  return result;
}

template <Basis BasisType, Quadrature QuadratureType>
void check_integration(const size_t min_pts, const size_t max_pts) noexcept {
  CAPTURE(BasisType);
  CAPTURE(QuadratureType);
  for (size_t num_pts = min_pts; num_pts < max_pts; ++num_pts) {
    CAPTURE(num_pts);
    const Mesh<1> mesh{num_pts, BasisType, QuadratureType};
    const auto logical_coords = logical_coordinates(mesh);
    DataVector u(num_pts);
    DataVector u_int_exact(num_pts);
    // Single term integrals
    for (size_t i = 0; i < num_pts - 1; ++i) {
      u = pow(get<0>(logical_coords), i);
      u_int_exact = 1.0 / (i + 1.0) * pow(get<0>(logical_coords), i + 1);
      u_int_exact -= 1.0 / (i + 1.0) * pow(-1.0, i + 1);
      const auto u_int_computed = integrate(u, mesh);
      CHECK_ITERABLE_APPROX(u_int_exact, u_int_computed);
    }
    // Full sum of terms
    u = 1.0;
    u_int_exact = get<0>(logical_coords);
    double u_int_exact_boundary = -1.0;
    for (size_t i = 1; i < num_pts - 1; ++i) {
      u += pow(get<0>(logical_coords), i);
      u_int_exact += 1.0 / (i + 1.0) * pow(get<0>(logical_coords), i + 1);
      u_int_exact_boundary += 1.0 / (i + 1.0) * pow(-1.0, i + 1);
    }
    u_int_exact -= u_int_exact_boundary;
    const auto u_int_computed = integrate(u, mesh);
    CHECK_ITERABLE_APPROX(u_int_exact, u_int_computed);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.NumericalAlgoriths.Spectral.IndefiniteIntegral",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  check_integration<Basis::Chebyshev, Quadrature::GaussLobatto>(
      2, maximum_number_of_points<Basis::Chebyshev>);
  check_integration<Basis::Chebyshev, Quadrature::Gauss>(
      2, maximum_number_of_points<Basis::Chebyshev>);
  check_integration<Basis::Legendre, Quadrature::GaussLobatto>(
      2, maximum_number_of_points<Basis::Legendre>);
  check_integration<Basis::Legendre, Quadrature::Gauss>(
      2, maximum_number_of_points<Basis::Legendre>);
}
}  // namespace Spectral
