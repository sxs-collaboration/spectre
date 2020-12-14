// Distributed under the MIT License.
// See LICENSE.txt for details.

// \file
// Tests of spectral operations that should work for any basis and quadrature.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Framework/TestCreation.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Math.hpp"

namespace {
void test_streaming() {
  CHECK(get_output(Spectral::Basis::Legendre) == "Legendre");
  CHECK(get_output(Spectral::Basis::Chebyshev) == "Chebyshev");
  CHECK(get_output(Spectral::Basis::FiniteDifference) == "FiniteDifference");

  CHECK(get_output(Spectral::Quadrature::Gauss) == "Gauss");
  CHECK(get_output(Spectral::Quadrature::GaussLobatto) == "GaussLobatto");
  CHECK(get_output(Spectral::Quadrature::CellCentered) == "CellCentered");
  CHECK(get_output(Spectral::Quadrature::FaceCentered) == "FaceCentered");
}

void test_creation() {
  CHECK(Spectral::Basis::Legendre ==
        TestHelpers::test_creation<Spectral::Basis>("Legendre"));
  CHECK(Spectral::Basis::Chebyshev ==
        TestHelpers::test_creation<Spectral::Basis>("Chebyshev"));
  CHECK(Spectral::Basis::FiniteDifference ==
        TestHelpers::test_creation<Spectral::Basis>("FiniteDifference"));

  CHECK(Spectral::Quadrature::Gauss ==
        TestHelpers::test_creation<Spectral::Quadrature>("Gauss"));
  CHECK(Spectral::Quadrature::GaussLobatto ==
        TestHelpers::test_creation<Spectral::Quadrature>("GaussLobatto"));
  CHECK(Spectral::Quadrature::CellCentered ==
        TestHelpers::test_creation<Spectral::Quadrature>("CellCentered"));
  CHECK(Spectral::Quadrature::FaceCentered ==
        TestHelpers::test_creation<Spectral::Quadrature>("FaceCentered"));
}

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
void test_exact_differentiation_impl(const Function& max_poly_deg) {
  INFO("Test exact differentiation.");
  CAPTURE(BasisType);
  CAPTURE(QuadratureType);
  for (size_t n = Spectral::minimum_number_of_points<BasisType, QuadratureType>;
       n <= Spectral::maximum_number_of_points<BasisType>; n++) {
    for (size_t p = 0; p <= max_poly_deg(n); p++) {
      const auto& collocation_pts =
          Spectral::collocation_points<BasisType, QuadratureType>(n);
      const auto& diff_matrix =
          Spectral::differentiation_matrix<BasisType, QuadratureType>(n);
      const auto u = unit_polynomial(p, collocation_pts);
      DataVector numeric_derivative{n};
      dgemv_('N', n, n, 1., diff_matrix.data(), diff_matrix.spacing(), u.data(),
             1, 0.0, numeric_derivative.data(), 1);
      const auto analytic_derivative =
          unit_polynomial_derivative(p, collocation_pts);
      CHECK_ITERABLE_APPROX(analytic_derivative, numeric_derivative);
    }
  }
}

template <Spectral::Basis BasisType, Spectral::Quadrature QuadratureType>
void test_weak_differentiation() {
  static_assert(BasisType == Spectral::Basis::Legendre,
                "test_weak_differentiation_matrix may not be correct for "
                "non-Legendre basis functions.");
  INFO("Test weak differentiation.");
  CAPTURE(BasisType);
  CAPTURE(QuadratureType);

  for (size_t n = Spectral::minimum_number_of_points<BasisType, QuadratureType>;
       n <= Spectral::maximum_number_of_points<BasisType>; n++) {
    CAPTURE(n);
    const DataVector& quad_weights =
        Spectral::quadrature_weights<BasisType, QuadratureType>(n);
    const Matrix& weak_diff_matrix =
        Spectral::weak_flux_differentiation_matrix<BasisType, QuadratureType>(
            n);
    const Matrix& diff_matrix =
        Spectral::differentiation_matrix<BasisType, QuadratureType>(n);
    Matrix expected_weak_diff_matrix(diff_matrix.rows(), diff_matrix.columns());

    for (size_t i = 0; i < n; ++i) {
      for (size_t l = 0; l < n; ++l) {
        expected_weak_diff_matrix(i, l) =
            quad_weights[l] / quad_weights[i] * diff_matrix(l, i);
      }
    }
    for (size_t i = 0; i < n; ++i) {
      for (size_t l = 0; l < n; ++l) {
        CAPTURE(i);
        CAPTURE(l);
        CHECK(expected_weak_diff_matrix(i, l) ==
              approx(weak_diff_matrix(i, l)));
      }
    }

    if (QuadratureType == Spectral::Quadrature::GaussLobatto) {
      for (size_t p = 0; p <= n - 1; p++) {
        const auto& collocation_pts =
            Spectral::collocation_points<BasisType, QuadratureType>(n);
        const auto u = unit_polynomial(p, collocation_pts);
        DataVector numeric_derivative{n};
        dgemv_('N', n, n, 1., weak_diff_matrix.data(),
               weak_diff_matrix.spacing(), u.data(), 1, 0.0,
               numeric_derivative.data(), 1);
        const auto analytic_derivative =
            unit_polynomial_derivative(p, collocation_pts);
        for (size_t i = 1; i < n - 1; ++i) {
          CHECK(numeric_derivative[i] == approx(-analytic_derivative[i]));
        }
      }
    }
  }
}

void test_exact_differentiation_matrices() {
  const auto minus_one = [](const size_t n) noexcept { return n - 1; };
  test_exact_differentiation_impl<Spectral::Basis::Legendre,
                                  Spectral::Quadrature::Gauss>(minus_one);
  test_exact_differentiation_impl<Spectral::Basis::Legendre,
                                  Spectral::Quadrature::GaussLobatto>(
      minus_one);
  test_exact_differentiation_impl<Spectral::Basis::Chebyshev,
                                  Spectral::Quadrature::Gauss>(minus_one);
  test_exact_differentiation_impl<Spectral::Basis::Chebyshev,
                                  Spectral::Quadrature::GaussLobatto>(
      minus_one);

  test_weak_differentiation<Spectral::Basis::Legendre,
                            Spectral::Quadrature::Gauss>();
  test_weak_differentiation<Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto>();
}

template <Spectral::Basis BasisType, Spectral::Quadrature QuadratureType>
void test_linear_filter_impl() {
  CAPTURE(BasisType);
  CAPTURE(QuadratureType);
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
    dgemv_('N', n, n, 1.0, filter_matrix.data(), filter_matrix.spacing(),
           u.data(), 1, 0.0, u_filtered.data(), 1);
    DataVector u_filtered_spectral(n);
    dgemv_('N', n, n, 1.0, nodal_to_modal_matrix.data(),
           nodal_to_modal_matrix.spacing(), u_filtered.data(), 1, 0.0,
           u_filtered_spectral.data(), 1);
    for (size_t s = 2; s < n; ++s) {
      CHECK(0.0 == approx(u_filtered_spectral[s]));
    }
  }
}

void test_linear_filter() {
  test_linear_filter_impl<Spectral::Basis::Legendre,
                          Spectral::Quadrature::Gauss>();
  test_linear_filter_impl<Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto>();
  test_linear_filter_impl<Spectral::Basis::Chebyshev,
                          Spectral::Quadrature::Gauss>();
  test_linear_filter_impl<Spectral::Basis::Chebyshev,
                          Spectral::Quadrature::GaussLobatto>();
}

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
             interp_matrix.spacing(), u.data(), 1, 0., interpolated_u.data(),
             1);
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
void test_exact_interpolation_impl(const Function& max_poly_deg) noexcept {
  const DataVector target_points{-0.5, -0.4837, 0.5, 0.9378, 1.};
  test_interpolation_matrix<BasisType, QuadratureType>(target_points,
                                                       max_poly_deg);
}

template <Spectral::Basis BasisType, Spectral::Quadrature QuadratureType,
          typename Function>
void test_exact_extrapolation_impl(const Function& max_poly_deg) {
  const DataVector target_points{-1.66, 1., 1.5, 1.98, 2.};
  // Errors are larger when extrapolating outside of the original grid:
  const double eps = 1.e-9;
  test_interpolation_matrix<BasisType, QuadratureType>(target_points,
                                                       max_poly_deg, eps);
}

void test_exact_extrapolation() {
  const auto minus_one = [](const size_t n) noexcept { return n - 1; };
  {
    INFO(
        "Legendre-Gauss interpolation is exact to polynomial order "
        "num_points-1");
    test_exact_interpolation_impl<Spectral::Basis::Legendre,
                                  Spectral::Quadrature::Gauss>(minus_one);
    test_exact_extrapolation_impl<Spectral::Basis::Legendre,
                                  Spectral::Quadrature::Gauss>(minus_one);
  }
  {
    INFO(
        "Legendre-Gauss-Lobatto interpolation is exact to polynomial "
        "order num_points-1");
    test_exact_interpolation_impl<Spectral::Basis::Legendre,
                                  Spectral::Quadrature::GaussLobatto>(
        minus_one);
    test_exact_extrapolation_impl<Spectral::Basis::Legendre,
                                  Spectral::Quadrature::GaussLobatto>(
        minus_one);
  }
  {
    INFO(
        "Chebyshev-Gauss interpolation is exact to polynomial "
        "order num_points-1");
    test_exact_interpolation_impl<Spectral::Basis::Chebyshev,
                                  Spectral::Quadrature::Gauss>(minus_one);
    test_exact_extrapolation_impl<Spectral::Basis::Chebyshev,
                                  Spectral::Quadrature::Gauss>(minus_one);
  }
  {
    INFO(
        "Chebyshev-Gauss-Lobatto interpolation is exact to polynomial "
        "order num_points-1");
    test_exact_interpolation_impl<Spectral::Basis::Chebyshev,
                                  Spectral::Quadrature::GaussLobatto>(
        minus_one);
    test_exact_extrapolation_impl<Spectral::Basis::Chebyshev,
                                  Spectral::Quadrature::GaussLobatto>(
        minus_one);
  }
}

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

void test_exact_quadrature() {
  {
    INFO(
        "Legendre-Gauss quadrature is exact to polynomial order "
        "2*num_points-1");
    test_exact_unit_weight_quadrature<Spectral::Basis::Legendre,
                                      Spectral::Quadrature::Gauss>(
        [](const size_t n) { return 2 * n - 1; });
  }
  {
    INFO(
        "Legendre-Gauss-Lobatto quadrature is exact to polynomial order "
        "2*num_points-3");
    test_exact_unit_weight_quadrature<Spectral::Basis::Legendre,
                                      Spectral::Quadrature::GaussLobatto>(
        [](const size_t n) { return 2 * n - 3; });
  }
  // For a function \f$f(x)\f$ the exact quadrature is
  // \f$\int_{-1}^{1}f(x)w(x)\mathrm{d}x\f$ where \f$w(x)=1/sqrt(1-x^2)\f$ is
  // the Chebyshev weight. We test this here for polynomials with unit
  // coefficients \f$f(x)=1+x+x^2+\ldots\f$.
  {
    INFO(
        "Chebyshev-Gauss quadrature is exact to polynomial order "
        "2*num_points-1");
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
  {
    INFO(
        "Chebyshev-Gauss-Lobatto quadrature is exact to polynomial order "
        "2*num_points-3");
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

template <Spectral::Basis BasisType, Spectral::Quadrature QuadratureType>
void test_quadrature_weights_impl() {
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

void test_quadrature_weights() {
  // Test that the Spectral::quadrature_weights are those used to compute
  // definite integrals, as opposed to the weighted inner product.
  test_quadrature_weights_impl<Spectral::Basis::Legendre,
                               Spectral::Quadrature::Gauss>();
  test_quadrature_weights_impl<Spectral::Basis::Legendre,
                               Spectral::Quadrature::GaussLobatto>();
  test_quadrature_weights_impl<Spectral::Basis::Chebyshev,
                               Spectral::Quadrature::Gauss>();
  test_quadrature_weights_impl<Spectral::Basis::Chebyshev,
                               Spectral::Quadrature::GaussLobatto>();
}

template <Spectral::Basis BasisType, Spectral::Quadrature QuadratureType>
void test_spectral_quantities_for_mesh_impl(const Mesh<1>& slice) {
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

void test_spectral_quantities_for_mesh() {
  /// [get_points_for_mesh]
  const Mesh<2> mesh2d{
      {{3, 4}},
      {{Spectral::Basis::Legendre, Spectral::Basis::Legendre}},
      {{Spectral::Quadrature::Gauss, Spectral::Quadrature::GaussLobatto}}};
  const auto collocation_points_in_first_dim =
      Spectral::collocation_points(mesh2d.slice_through(0));
  /// [get_points_for_mesh]
  test_spectral_quantities_for_mesh_impl<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::Gauss>(
      mesh2d.slice_through(0));
  test_spectral_quantities_for_mesh_impl<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::GaussLobatto>(
      mesh2d.slice_through(1));
}

void test_gauss_points_boundary_interpolation_and_lifting() noexcept {
  const auto max_poly_deg = [](const size_t num_pts) noexcept {
    return num_pts - 1;
  };
  constexpr Spectral::Basis BasisType = Spectral::Basis::Legendre;
  constexpr Spectral::Quadrature QuadratureType = Spectral::Quadrature::Gauss;

  DataVector interpolated_u_lower(1, 0.);
  DataVector interpolated_u_upper(1, 0.);
  for (size_t n = Spectral::minimum_number_of_points<BasisType, QuadratureType>;
       n < Spectral::maximum_number_of_points<BasisType>; n++) {
    CAPTURE(n);
    const auto& collocation_pts =
        Spectral::collocation_points<BasisType, QuadratureType>(n);
    Mesh<1> local_mesh{n, BasisType, QuadratureType};
    const std::pair<Matrix, Matrix>& boundary_interp_matrices =
        Spectral::boundary_interpolation_matrices(local_mesh);
    const std::pair<DataVector, DataVector>& lifting_terms =
        Spectral::boundary_lifting_term(local_mesh);
    const DataVector& quad_weights = Spectral::quadrature_weights(local_mesh);
    for (size_t p = 0; p <= max_poly_deg(n); p++) {
      CAPTURE(p);
      const DataVector u = unit_polynomial(p, collocation_pts);

      dgemv_('n', 1, n, 1., boundary_interp_matrices.first.data(),
             boundary_interp_matrices.first.spacing(), u.data(), 1, 0.,
             interpolated_u_lower.data(), 1);
      CHECK_ITERABLE_APPROX(unit_polynomial(p, DataVector{-1.0}),
                            interpolated_u_lower);
      dgemv_('n', 1, n, 1., boundary_interp_matrices.second.data(),
             boundary_interp_matrices.second.spacing(), u.data(), 1, 0.,
             interpolated_u_upper.data(), 1);
      CHECK_ITERABLE_APPROX(unit_polynomial(p, DataVector{1.0}),
                            interpolated_u_upper);

      DataVector lift_lagrange_lower(u.size());
      for (size_t i = 0; i < lift_lagrange_lower.size(); ++i) {
        lift_lagrange_lower[i] = boundary_interp_matrices.first(0, i);
      }
      CHECK_ITERABLE_APPROX(DataVector{lift_lagrange_lower / quad_weights},
                            lifting_terms.first);

      DataVector lift_lagrange_upper(u.size());
      for (size_t i = 0; i < lift_lagrange_upper.size(); ++i) {
        lift_lagrange_upper[i] = boundary_interp_matrices.second(0, i);
      }
      CHECK_ITERABLE_APPROX(DataVector{lift_lagrange_upper / quad_weights},
                            lifting_terms.second);
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Spectral",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  test_streaming();
  test_creation();
  test_exact_differentiation_matrices();
  test_linear_filter();
  test_exact_extrapolation();
  test_exact_quadrature();
  test_quadrature_weights();
  test_spectral_quantities_for_mesh();
  test_gauss_points_boundary_interpolation_and_lifting();
}
