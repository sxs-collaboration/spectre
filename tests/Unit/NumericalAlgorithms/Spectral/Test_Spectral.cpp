// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/Spectral/LegendreGaussLobatto.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ConstantExpressions.hpp"

SPECTRE_TEST_CASE(
    "Unit.Numerical.Spectral.LegendreGaussLobatto.LinearFilterMatrix",
    "[NumericalAlgorithms][Spectral][Unit]") {
  for (size_t n = 2; n < 10; ++n) {
    const Matrix& filter_matrix = Basis::lgl::linear_filter_matrix(n);
    const Matrix& grid_points_to_spectral_matrix =
        Basis::lgl::grid_points_to_spectral_matrix(n);
    const DataVector& collocation_points = Basis::lgl::collocation_points(n);
    DataVector u(n);
    for (size_t s = 0; s < n; ++s) {
      u[s] = exp(collocation_points[s]);
    }
    DataVector u_filtered(n);
    dgemv_('N', n, n, 1.0, filter_matrix.data(), n, u.data(), 1, 0.0,
           u_filtered.data(), 1);
    DataVector u_spectral(n);
    dgemv_('N', n, n, 1.0, grid_points_to_spectral_matrix.data(), n,
           u_filtered.data(), 1, 0.0, u_spectral.data(), 1);
    for (size_t s = 2; s < n; ++s) {
      CHECK(0.0 == approx(u_spectral[s]));
    }
  }
}

SPECTRE_TEST_CASE(
    "Unit.Numerical.Spectral.LegendreGaussLobatto.InterpolationMatrix",
    "[NumericalAlgorithms][Spectral][Unit]") {
  auto check_interp = [](const size_t num_pts, auto func) {
    const DataVector& collocation_points =
        Basis::lgl::collocation_points(num_pts);
    DataVector u(num_pts);
    for (size_t i = 0; i < num_pts; ++i) {
      u[i] = func(collocation_points[i]);
    }
    DataVector new_points{-0.5, -0.4837, 0.5, 0.9378, 1.0};
    DataVector interpolated_u(new_points.size(), 0.0);

    const Matrix interp_matrix =
        Basis::lgl::interpolation_matrix(num_pts, new_points);
    dgemv_('n', new_points.size(), num_pts, 1.0, interp_matrix.data(),
           new_points.size(), u.data(), 1, 0.0, interpolated_u.data(), 1);

    CHECK(interpolated_u.size() == new_points.size());
    for (size_t i = 0; i < new_points.size(); ++i) {
      CHECK(func(new_points[i]) == approx(interpolated_u[i]));
    }
  };

  check_interp(2, [](const double x) { return x + 1.0; });
  check_interp(3, [](const double x) { return x * x + x + 1.0; });
  check_interp(4,
               [](const double x) { return pow<3>(x) + pow<2>(x) + x + 1.0; });
  check_interp(5, [](const double x) {
    return pow<4>(x) + pow<3>(x) + pow<2>(x) + x + 1.0;
  });
  check_interp(6, [](const double x) {
    return pow<5>(x) + pow<4>(x) + pow<3>(x) + pow<2>(x) + x + 1.0;
  });
  check_interp(7, [](const double x) {
    return pow<6>(x) + pow<5>(x) + pow<4>(x) + pow<3>(x) + pow<2>(x) + x + 1.0;
  });
  check_interp(8, [](const double x) {
    return pow<7>(x) + pow<6>(x) + pow<5>(x) + pow<4>(x) + pow<3>(x) +
           pow<2>(x) + x + 1.0;
  });
  check_interp(9, [](const double x) {
    return pow<8>(x) + pow<7>(x) + pow<6>(x) + pow<5>(x) + pow<4>(x) +
           pow<3>(x) + pow<2>(x) + x + 1.0;
  });
}
