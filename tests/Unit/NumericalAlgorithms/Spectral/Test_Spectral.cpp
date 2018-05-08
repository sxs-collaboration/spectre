// Distributed under the MIT License.
// See LICENSE.txt for details.

// \file
// Tests of spectral operations that should work for any basis and quadrature.

#include "tests/Unit/TestingFramework.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Blas.hpp"

namespace {

template <Spectral::Basis BasisType, Spectral::Quadrature QuadratureType>
void test_linear_filter(const size_t num_points) {
  const Matrix& filter_matrix =
      Spectral::linear_filter_matrix<BasisType, QuadratureType>(num_points);
  const Matrix& grid_points_to_spectral_matrix =
      Spectral::grid_points_to_spectral_matrix<BasisType, QuadratureType>(
          num_points);
  const DataVector& collocation_pts =
      Spectral::collocation_points<BasisType, QuadratureType>(num_points);
  const DataVector u = exp(collocation_pts);
  DataVector u_filtered(num_points);
  dgemv_('N', num_points, num_points, 1.0, filter_matrix.data(), num_points,
         u.data(), 1, 0.0, u_filtered.data(), 1);
  DataVector u_filtered_spectral(num_points);
  dgemv_('N', num_points, num_points, 1.0,
         grid_points_to_spectral_matrix.data(), num_points, u_filtered.data(),
         1, 0.0, u_filtered_spectral.data(), 1);
  for (size_t s = 2; s < num_points; ++s) {
    CHECK(0.0 == approx(u_filtered_spectral[s]));
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.LinearFilter",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  SECTION("Legendre-Gauss-Lobatto") {
    for (size_t n = Spectral::minimum_number_of_points<
             Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto>;
         n <= Spectral::maximum_number_of_points<Spectral::Basis::Legendre>;
         ++n) {
      test_linear_filter<Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto>(n);
    }
  }
}

namespace {

template <Spectral::Basis BasisType, Spectral::Quadrature QuadratureType>
void test_exact_interpolation(size_t num_pts, int poly_deg) {
  const DataVector& collocation_pts =
      Spectral::collocation_points<BasisType, QuadratureType>(num_pts);
  auto polynomial = [poly_deg](const DataVector& x) {
    auto func_value = DataVector(x.size(), 1.);
    for (int p = 1; p <= poly_deg; p++) {
      func_value += pow(x, p);
    }
    return func_value;
  };
  const DataVector u = polynomial(collocation_pts);
  const DataVector new_points{-0.5, -0.4837, 0.5, 0.9378, 1.0};
  DataVector interpolated_u(new_points.size(), 0.0);
  const Matrix interp_matrix =
      Spectral::interpolation_matrix<BasisType, QuadratureType>(num_pts,
                                                                new_points);
  dgemv_('n', new_points.size(), num_pts, 1.0, interp_matrix.data(),
         new_points.size(), u.data(), 1, 0.0, interpolated_u.data(), 1);
  CHECK(interpolated_u.size() == new_points.size());
  CHECK_ITERABLE_APPROX(polynomial(new_points), interpolated_u);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.ExactInterpolation",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  SECTION(
      "Legendre-Gauss-Lobatto interpolation is exact to polynomial "
      "order num_points-1") {
    for (size_t n = Spectral::minimum_number_of_points<
             Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto>;
         n <= Spectral::maximum_number_of_points<Spectral::Basis::Legendre>;
         n++) {
      for (size_t p = 0; p <= n - 1; p++) {
        test_exact_interpolation<Spectral::Basis::Legendre,
                                 Spectral::Quadrature::GaussLobatto>(n, p);
      }
    }
  }
}

namespace {

template <Spectral::Basis BasisType, Spectral::Quadrature QuadratureType>
void test_exact_quadrature(size_t num_pts, int poly_deg) {
  const DataVector& collocation_pts =
      Spectral::collocation_points<BasisType, QuadratureType>(num_pts);
  const DataVector& weights =
      Spectral::quadrature_weights<BasisType, QuadratureType>(num_pts);
  auto polynomial = [poly_deg](const DataVector& x) {
    auto func_value = DataVector(x.size(), 1.);
    for (int p = 1; p <= poly_deg; p++) {
      func_value += pow(x, p);
    }
    return func_value;
  };
  auto polynomial_integral = [poly_deg](const double x) {
    double func_value = 0.;
    for (int p = 1; p <= poly_deg + 1; p++) {
      func_value += pow(x, p) / double(p);
    }
    return func_value;
  };
  const DataVector u = polynomial(collocation_pts);
  const double numeric_integral =
      ddot_(num_pts, u.data(), 1, weights.data(), 1);
  const double analytic_integral =
      polynomial_integral(1.) - polynomial_integral(-1.);
  CHECK_ITERABLE_APPROX(analytic_integral, numeric_integral);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.ExactQuadrature",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  SECTION(
      "Legendre-Gauss-Lobatto quadrature is exact to polynomial order "
      "2*num_points-3") {
    for (size_t n = Spectral::minimum_number_of_points<
             Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto>;
         n <= Spectral::maximum_number_of_points<Spectral::Basis::Legendre>;
         n++) {
      for (size_t p = 0; p <= 2 * n - 3; p++) {
        test_exact_quadrature<Spectral::Basis::Legendre,
                              Spectral::Quadrature::GaussLobatto>(n, p);
      }
    }
  }
}
