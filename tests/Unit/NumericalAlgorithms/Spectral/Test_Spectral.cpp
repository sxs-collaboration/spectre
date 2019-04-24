// Distributed under the MIT License.
// See LICENSE.txt for details.

// \file
// Tests of spectral operations that should work for any basis and quadrature.

#include "tests/Unit/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Math.hpp"

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.streaming",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  CHECK(get_output(Spectral::Basis::Legendre) == "Legendre");

  CHECK(get_output(Spectral::Quadrature::Gauss) == "Gauss");
  CHECK(get_output(Spectral::Quadrature::GaussLobatto) == "GaussLobatto");
}

namespace {

DataVector unit_polynomial(const size_t deg, const DataVector& x) {
  // Choose all polynomial coefficients to be one
  const std::vector<double> coeffs(deg + 1, 1.);
  return evaluate_polynomial(coeffs, x);
}
DataVector unit_polynomial_derivative(const size_t deg, const DataVector& x) {
  std::vector<double> coeffs(deg);
  for (size_t p = 0; p < coeffs.size(); p++) {
    coeffs[p] = p + 1;
  }
  return evaluate_polynomial(coeffs, x);
}
double unit_polynomial_integral(const size_t deg) {
  std::vector<double> coeffs(deg + 2);
  coeffs[0] = 0.;
  for (size_t p = 1; p < coeffs.size(); p++) {
    coeffs[p] = 1. / p;
  }
  const auto integrals = evaluate_polynomial(coeffs, DataVector{-1., 1.});
  return integrals[1] - integrals[0];
}

template <Spectral::Basis BasisType, Spectral::Quadrature QuadratureType,
          typename Function>
void test_exact_differentiation(const Function& max_poly_deg) {
  for (size_t n = Spectral::minimum_number_of_points<BasisType, QuadratureType>;
       n <= Spectral::maximum_number_of_points<BasisType>; n++) {
    for (size_t p = 0; p <= max_poly_deg(n); p++) {
      const auto& collocation_pts =
          Spectral::collocation_points<BasisType, QuadratureType>(n);
      const auto& diff_matrix =
          Spectral::differentiation_matrix<BasisType, QuadratureType>(n);
      const auto u = unit_polynomial(p, collocation_pts);
      DataVector numeric_derivative{n};
      dgemv_('N', n, n, 1., diff_matrix.data(), n, u.data(), 1, 0.0,
             numeric_derivative.data(), 1);
      const auto analytic_derivative =
          unit_polynomial_derivative(p, collocation_pts);
      CHECK_ITERABLE_APPROX(analytic_derivative, numeric_derivative);
    }
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.ExactDifferentiation",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  const auto minus_one = [](const size_t n) noexcept { return n - 1; };
  SECTION(
      "Legendre-Gauss differentiation is exact to polynomial order "
      "num_points - 1") {
    test_exact_differentiation<Spectral::Basis::Legendre,
                               Spectral::Quadrature::Gauss>(minus_one);
  }
  SECTION(
      "Legendre-Gauss-Lobatto differentiation is exact to polynomial order "
      "num_points - 1") {
    test_exact_differentiation<Spectral::Basis::Legendre,
                               Spectral::Quadrature::GaussLobatto>(minus_one);
  }
  SECTION(
      "Chebyshev-Gauss differentiation is exact to polynomial order "
      "num_points - 1") {
    test_exact_differentiation<Spectral::Basis::Chebyshev,
                               Spectral::Quadrature::Gauss>(minus_one);
  }
  SECTION(
      "Chebyshev-Gauss-Lobatto differentiation is exact to polynomial order "
      "num_points - 1") {
    test_exact_differentiation<Spectral::Basis::Chebyshev,
                               Spectral::Quadrature::GaussLobatto>(minus_one);
  }
}

namespace {

template <Spectral::Basis BasisType, Spectral::Quadrature QuadratureType>
void test_linear_filter() {
  for (size_t n = Spectral::minimum_number_of_points<BasisType, QuadratureType>;
       n <= Spectral::maximum_number_of_points<BasisType>; n++) {
    const auto& filter_matrix =
        Spectral::linear_filter_matrix<BasisType, QuadratureType>(n);
    const auto& nodal_to_modal_matrix =
        Spectral::nodal_to_modal_matrix<BasisType, QuadratureType>(n);
    const auto& collocation_pts =
        Spectral::collocation_points<BasisType, QuadratureType>(n);
    const DataVector u = exp(collocation_pts);
    DataVector u_filtered(n);
    dgemv_('N', n, n, 1.0, filter_matrix.data(), n, u.data(), 1, 0.0,
           u_filtered.data(), 1);
    DataVector u_filtered_spectral(n);
    dgemv_('N', n, n, 1.0, nodal_to_modal_matrix.data(), n, u_filtered.data(),
           1, 0.0, u_filtered_spectral.data(), 1);
    for (size_t s = 2; s < n; ++s) {
      CHECK(0.0 == approx(u_filtered_spectral[s]));
    }
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.LinearFilter",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  SECTION("Legendre-Gauss") {
    test_linear_filter<Spectral::Basis::Legendre,
                       Spectral::Quadrature::Gauss>();
  }
  SECTION("Legendre-Gauss-Lobatto") {
    test_linear_filter<Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto>();
  }
  SECTION("Chebyshev-Gauss") {
    test_linear_filter<Spectral::Basis::Chebyshev,
                       Spectral::Quadrature::Gauss>();
  }
  SECTION("Chebyshev-Gauss-Lobatto") {
    test_linear_filter<Spectral::Basis::Chebyshev,
                       Spectral::Quadrature::GaussLobatto>();
  }
}

namespace {

// By default, uses the default tolerance for floating-point comparisons. When
// a non-zero eps is passed in as last argument, uses that tolerance instead.
template <Spectral::Basis BasisType, Spectral::Quadrature QuadratureType,
          typename Function>
void test_interpolation_matrix(const DataVector& target_points,
                               const Function& max_poly_deg,
                               const double eps = 0.) noexcept {
  DataVector interpolated_u(target_points.size(), 0.);
  for (size_t n = Spectral::minimum_number_of_points<BasisType, QuadratureType>;
       n <= Spectral::maximum_number_of_points<BasisType>; n++) {
    const auto& collocation_pts =
        Spectral::collocation_points<BasisType, QuadratureType>(n);
    const auto interp_matrix =
        Spectral::interpolation_matrix<BasisType, QuadratureType>(
            n, target_points);
    for (size_t p = 0; p <= max_poly_deg(n); p++) {
      const DataVector u = unit_polynomial(p, collocation_pts);
      dgemv_('n', target_points.size(), n, 1., interp_matrix.data(),
             target_points.size(), u.data(), 1, 0., interpolated_u.data(), 1);
      CHECK(interpolated_u.size() == target_points.size());
      if (eps <= 0.) {
        CHECK_ITERABLE_APPROX(unit_polynomial(p, target_points),
                              interpolated_u);
      } else {
        Approx local_approx = Approx::custom().epsilon(eps).scale(1.);
        CHECK_ITERABLE_CUSTOM_APPROX(unit_polynomial(p, target_points),
                                     interpolated_u, local_approx);
      }
    }
  }
}

template <Spectral::Basis BasisType, Spectral::Quadrature QuadratureType,
          typename Function>
void test_exact_interpolation(const Function& max_poly_deg) noexcept {
  const DataVector target_points{-0.5, -0.4837, 0.5, 0.9378, 1.};
  test_interpolation_matrix<BasisType, QuadratureType>(target_points,
                                                       max_poly_deg);
}

template <Spectral::Basis BasisType, Spectral::Quadrature QuadratureType,
          typename Function>
void test_exact_extrapolation(const Function& max_poly_deg) {
  const DataVector target_points{-1.66, 1., 1.5, 1.98, 2.};
  // Errors are larger when extrapolating outside of the original grid:
  const double eps = 1.e-9;
  test_interpolation_matrix<BasisType, QuadratureType>(target_points,
                                                       max_poly_deg, eps);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.ExactInterpolation",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  const auto minus_one = [](const size_t n) noexcept { return n - 1; };
  SECTION(
      "Legendre-Gauss interpolation is exact to polynomial order "
      "num_points-1") {
    test_exact_interpolation<Spectral::Basis::Legendre,
                             Spectral::Quadrature::Gauss>(minus_one);
    test_exact_extrapolation<Spectral::Basis::Legendre,
                             Spectral::Quadrature::Gauss>(minus_one);
  }
  SECTION(
      "Legendre-Gauss-Lobatto interpolation is exact to polynomial "
      "order num_points-1") {
    test_exact_interpolation<Spectral::Basis::Legendre,
                             Spectral::Quadrature::GaussLobatto>(minus_one);
    test_exact_extrapolation<Spectral::Basis::Legendre,
                             Spectral::Quadrature::GaussLobatto>(minus_one);
  }
  SECTION(
      "Chebyshev-Gauss interpolation is exact to polynomial "
      "order num_points-1") {
    test_exact_interpolation<Spectral::Basis::Chebyshev,
                             Spectral::Quadrature::Gauss>(minus_one);
    test_exact_extrapolation<Spectral::Basis::Chebyshev,
                             Spectral::Quadrature::Gauss>(minus_one);
  }
  SECTION(
      "Chebyshev-Gauss-Lobatto interpolation is exact to polynomial "
      "order num_points-1") {
    test_exact_interpolation<Spectral::Basis::Chebyshev,
                             Spectral::Quadrature::GaussLobatto>(minus_one);
    test_exact_extrapolation<Spectral::Basis::Chebyshev,
                             Spectral::Quadrature::GaussLobatto>(minus_one);
  }
}

namespace {

template <Spectral::Basis BasisType, Spectral::Quadrature QuadratureType>
void test_exact_quadrature(const size_t n, const size_t p,
                           const double analytic_quadrature) {
  const auto& collocation_pts =
      Spectral::collocation_points<BasisType, QuadratureType>(n);
  // Get the \f$w_k\f$, as opposed to the Spectral::quadrature_weights that are
  // used to compute definite integrals (see test below).
  const auto w_k =
      Spectral::compute_collocation_points_and_weights<BasisType,
                                                       QuadratureType>(n)
          .second;
  const DataVector u = unit_polynomial(p, collocation_pts);
  const double numeric_quadrature = ddot_(n, u.data(), 1, w_k.data(), 1);
  CHECK_ITERABLE_APPROX(analytic_quadrature, numeric_quadrature);
}

template <Spectral::Basis BasisType, Spectral::Quadrature QuadratureType,
          typename Function>
void test_exact_unit_weight_quadrature(const Function& max_poly_deg) {
  for (size_t n = Spectral::minimum_number_of_points<BasisType, QuadratureType>;
       n <= Spectral::maximum_number_of_points<BasisType>; n++) {
    for (size_t p = 0; p <= max_poly_deg(n); p++) {
      const double analytic_quadrature = unit_polynomial_integral(p);
      test_exact_quadrature<BasisType, QuadratureType>(n, p,
                                                       analytic_quadrature);
    }
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.ExactQuadrature",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  SECTION(
      "Legendre-Gauss quadrature is exact to polynomial order 2*num_points-1") {
    test_exact_unit_weight_quadrature<Spectral::Basis::Legendre,
                                      Spectral::Quadrature::Gauss>(
        [](const size_t n) { return 2 * n - 1; });
  }
  SECTION(
      "Legendre-Gauss-Lobatto quadrature is exact to polynomial order "
      "2*num_points-3") {
    test_exact_unit_weight_quadrature<Spectral::Basis::Legendre,
                                      Spectral::Quadrature::GaussLobatto>(
        [](const size_t n) { return 2 * n - 3; });
  }
  // For a function \f$f(x)\f$ the exact quadrature is
  // \f$\int_{-1}^{1}f(x)w(x)\mathrm{d}x\f$ where \f$w(x)=1/sqrt(1-x^2)\f$ is
  // the Chebyshev weight. We test this here for polynomials with unit
  // coefficients \f$f(x)=1+x+x^2+\ldots\f$.
  SECTION(
      "Chebyshev-Gauss quadrature is exact to polynomial order "
      "2*num_points-1") {
    test_exact_quadrature<Spectral::Basis::Chebyshev,
                          Spectral::Quadrature::Gauss>(1, 1, M_PI);
    test_exact_quadrature<Spectral::Basis::Chebyshev,
                          Spectral::Quadrature::Gauss>(2, 3, 3. * M_PI / 2.);
    test_exact_quadrature<Spectral::Basis::Chebyshev,
                          Spectral::Quadrature::Gauss>(3, 5, 15. * M_PI / 8.);
    test_exact_quadrature<Spectral::Basis::Chebyshev,
                          Spectral::Quadrature::Gauss>(4, 7, 35. * M_PI / 16.);
    test_exact_quadrature<Spectral::Basis::Chebyshev,
                          Spectral::Quadrature::Gauss>(5, 9,
                                                       315. * M_PI / 128.);
    test_exact_quadrature<Spectral::Basis::Chebyshev,
                          Spectral::Quadrature::Gauss>(6, 11,
                                                       693. * M_PI / 256.);
  }
  SECTION(
      "Chebyshev-Gauss-Lobatto quadrature is exact to polynomial order "
      "2*num_points-3") {
    test_exact_quadrature<Spectral::Basis::Chebyshev,
                          Spectral::Quadrature::GaussLobatto>(2, 1, M_PI);
    test_exact_quadrature<Spectral::Basis::Chebyshev,
                          Spectral::Quadrature::GaussLobatto>(3, 3,
                                                              3. * M_PI / 2.);
    test_exact_quadrature<Spectral::Basis::Chebyshev,
                          Spectral::Quadrature::GaussLobatto>(4, 5,
                                                              15. * M_PI / 8.);
    test_exact_quadrature<Spectral::Basis::Chebyshev,
                          Spectral::Quadrature::GaussLobatto>(5, 7,
                                                              35. * M_PI / 16.);
    test_exact_quadrature<Spectral::Basis::Chebyshev,
                          Spectral::Quadrature::GaussLobatto>(
        6, 9, 315. * M_PI / 128.);
  }
}

namespace {

template <Spectral::Basis BasisType, Spectral::Quadrature QuadratureType>
void test_quadrature_weights() {
  for (size_t n = Spectral::minimum_number_of_points<BasisType, QuadratureType>;
       n <= Spectral::maximum_number_of_points<BasisType>; n++) {
    const auto& weights =
        Spectral::quadrature_weights<BasisType, QuadratureType>(n);
    const auto w_k =
        Spectral::compute_collocation_points_and_weights<BasisType,
                                                         QuadratureType>(n)
            .second;
    const auto& collocation_pts =
        Spectral::collocation_points<BasisType, QuadratureType>(n);
    const auto inverse_weight_function_values =
        Spectral::compute_inverse_weight_function_values<BasisType>(
            collocation_pts);
    CHECK_ITERABLE_APPROX(weights, w_k * inverse_weight_function_values);
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.QuadratureWeights",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  // Test that the Spectral::quadrature_weights are those used to compute
  // definite integrals, as opposed to the weighted inner product.
  test_quadrature_weights<Spectral::Basis::Legendre,
                          Spectral::Quadrature::Gauss>();
  test_quadrature_weights<Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto>();
  test_quadrature_weights<Spectral::Basis::Chebyshev,
                          Spectral::Quadrature::Gauss>();
  test_quadrature_weights<Spectral::Basis::Chebyshev,
                          Spectral::Quadrature::GaussLobatto>();
}

namespace {

template <Spectral::Basis BasisType, Spectral::Quadrature QuadratureType>
void test_spectral_quantities_for_mesh(const Mesh<1>& slice) {
  const auto num_points = slice.extents(0);
  const auto& expected_points =
      Spectral::collocation_points<BasisType, QuadratureType>(num_points);
  CHECK(Spectral::collocation_points(slice) == expected_points);
  const auto& expected_weights =
      Spectral::quadrature_weights<BasisType, QuadratureType>(num_points);
  CHECK(Spectral::quadrature_weights(slice) == expected_weights);
  const auto& expected_diff_matrix =
      Spectral::differentiation_matrix<BasisType, QuadratureType>(num_points);
  CHECK(Spectral::differentiation_matrix(slice) == expected_diff_matrix);
  const DataVector target_points{-0.5, -0.1, 0., 0.75, 0.9888, 1.};
  const auto expected_interp_matrix_points =
      Spectral::interpolation_matrix<BasisType, QuadratureType>(num_points,
                                                                target_points);
  CHECK(Spectral::interpolation_matrix(slice, target_points) ==
        expected_interp_matrix_points);
  const auto& expected_vand_matrix =
      Spectral::modal_to_nodal_matrix<BasisType, QuadratureType>(num_points);
  CHECK(Spectral::modal_to_nodal_matrix(slice) == expected_vand_matrix);
  const auto& expected_inv_vand_matrix =
      Spectral::nodal_to_modal_matrix<BasisType, QuadratureType>(num_points);
  CHECK(Spectral::nodal_to_modal_matrix(slice) == expected_inv_vand_matrix);
  const auto& expected_lin_matrix =
      Spectral::linear_filter_matrix<BasisType, QuadratureType>(num_points);
  CHECK(Spectral::linear_filter_matrix(slice) == expected_lin_matrix);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.Mesh",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  /// [get_points_for_mesh]
  const Mesh<2> mesh2d{
      {{3, 4}},
      {{Spectral::Basis::Legendre, Spectral::Basis::Legendre}},
      {{Spectral::Quadrature::Gauss, Spectral::Quadrature::GaussLobatto}}};
  const auto collocation_points_in_first_dim =
      Spectral::collocation_points(mesh2d.slice_through(0));
  /// [get_points_for_mesh]
  test_spectral_quantities_for_mesh<Spectral::Basis::Legendre,
                                    Spectral::Quadrature::Gauss>(
      mesh2d.slice_through(0));
  test_spectral_quantities_for_mesh<Spectral::Basis::Legendre,
                                    Spectral::Quadrature::GaussLobatto>(
      mesh2d.slice_through(1));
}
