// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Helpers/NumericalAlgorithms/Spectral/PolynomialTestFunctions.hpp"

#include <cmath>
#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/ApplyMatrix.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/DifferentiationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/IntegrationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/InterpolationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/MaximumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/MinimumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/ModalToNodalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/NodalToModalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/QuadratureWeights.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace PolynomialTestFunctions {

namespace {
void test_definite_integral(const DataVector& u, const DataVector& weights,
                            const Monomial& f) {
  const double result = ddot_(u.size(), weights.data(), 1, u.data(), 1);
  CHECK(f.definite_integral() == approx(result));
}

void test_derivative(const DataVector& u, const Matrix& m, const Monomial& f,
                     const DataVector& xi) {
  const auto custom_approx = Approx::custom().epsilon(1.0e-13).scale(1.0);
  const DataVector expected_du = f.df_dx(xi);
  const DataVector du = apply_matrix(m, u);
  CHECK_ITERABLE_CUSTOM_APPROX(du, expected_du, custom_approx);
}

void test_indefinite_integral(const DataVector& u, const Matrix& m,
                              const Monomial& f, const DataVector& xi) {
  const DataVector expected_int_u = f.int_f(xi);
  const DataVector int_u = apply_matrix(m, u);
  CHECK_ITERABLE_APPROX(int_u, expected_int_u);
}

void test_interpolation(const DataVector& u, const DataVector& xi_target,
                        const Matrix& m, const Monomial& f) {
  const DataVector expected_u = f(xi_target);
  const DataVector u_target = apply_matrix(m, u);
  CHECK_ITERABLE_APPROX(u_target, expected_u);
}

template <Spectral::Basis basis>
void test_transforms(const DataVector& u, const Matrix& modal_to_nodal_matrix,
                     const Matrix& nodal_to_modal_matrix, const Monomial& f) {
  const DataVector expected_u_k = f.modes<basis>();
  const DataVector u_k = apply_matrix(nodal_to_modal_matrix, u);
  for (size_t i = 0; i < expected_u_k.size(); ++i) {
    CHECK(u_k[i] == approx(expected_u_k[i]));
  }
  const DataVector transformed_u = apply_matrix(modal_to_nodal_matrix, u_k);
  CHECK_ITERABLE_APPROX(u, transformed_u);
}
}  // namespace

template <Spectral::Basis basis, Spectral::Quadrature quadrature>
void test_orthogonal_polynomial() {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> xi_distribution(-1.0, 1.0);
  const auto xi_target = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&xi_distribution), 5_st);
  for (size_t n = Spectral::minimum_number_of_points<basis, quadrature>;
       n <= Spectral::maximum_number_of_points<basis>; ++n) {
    const DataVector& xi = Spectral::collocation_points<basis, quadrature>(n);
    const Matrix& dm = Spectral::differentiation_matrix<basis, quadrature>(n);
    const DataVector& integration_weights =
        Spectral::quadrature_weights<basis, quadrature>(n);
    const Matrix& interp_m =
        Spectral::interpolation_matrix<basis, quadrature>(n, xi_target);
    const Matrix& int_m = Spectral::integration_matrix<basis, quadrature>(n);
    const Matrix& m_to_n =
        Spectral::modal_to_nodal_matrix<basis, quadrature>(n);
    const Matrix& n_to_m =
        Spectral::nodal_to_modal_matrix<basis, quadrature>(n);
    for (size_t pow_x = 0; pow_x < n; ++pow_x) {
      const Monomial f{pow_x};
      const DataVector u = f(xi);
      test_definite_integral(u, integration_weights, f);
      if (pow_x < n - 1) {
        test_indefinite_integral(u, int_m, f, xi);
      }
      test_interpolation(u, xi_target, interp_m, f);
      test_derivative(u, dm, f, xi);
      test_transforms<basis>(u, m_to_n, n_to_m, f);
    }
  }
}

template void test_orthogonal_polynomial<Spectral::Basis::Chebyshev,
                                         Spectral::Quadrature::GaussLobatto>();
template void test_orthogonal_polynomial<Spectral::Basis::Chebyshev,
                                         Spectral::Quadrature::Gauss>();
template void test_orthogonal_polynomial<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::GaussLobatto>();
template void test_orthogonal_polynomial<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::Gauss>();

Monomial::Monomial(const size_t pow_x) : pow_x_(pow_x) {}

DataVector Monomial::operator()(const DataVector& x) const {
  return pow(x, static_cast<double>(pow_x_));
}

double Monomial::operator()(const double x) const {
  return pow(x, static_cast<double>(pow_x_));
}

DataVector Monomial::df_dx(const DataVector& x) const {
  if (pow_x_ == 0) {
    return DataVector{x.size(), 0.0};
  }
  return static_cast<double>(pow_x_) * pow(x, static_cast<double>(pow_x_ - 1));
}

DataVector Monomial::int_f(const DataVector& x) const {
  const auto n_plus_one = static_cast<double>(pow_x_ + 1);
  const double constant =
      pow_x_ % 2 == 0 ? 1.0 / n_plus_one : -1.0 / n_plus_one;
  return pow(x, n_plus_one) / n_plus_one + constant;
}

double Monomial::definite_integral() const {
  const auto n_plus_one = static_cast<double>(pow_x_ + 1);
  return pow_x_ % 2 == 0 ? 2.0 / n_plus_one : 0.0;
}

// See, for example, Table of Integrals, Series, and Products 8.922
template <>
DataVector Monomial::modes<Spectral::Basis::Legendre>() const {
  DataVector result{pow_x_ + 1, 0.0};
  const size_t n = pow_x_ / 2;
  if (pow_x_ % 2 == 0) {
    double coeff = 1.0 / (2.0 * static_cast<double>(n) + 1.0);
    result[0] = coeff;
    for (size_t k = 1; k <= pow_x_ / 2; ++k) {
      coeff *=
          2.0 * (static_cast<double>(n) - static_cast<double>(k) + 1.0) /
          (2.0 * static_cast<double>(n) + 2.0 * static_cast<double>(k) + 1.0);
      result[2 * k] = coeff * (4.0 * static_cast<double>(k) + 1.0);
    }
  } else {
    double coeff = 1.0 / (2.0 * static_cast<double>(n) + 3.0);
    result[1] = 3.0 * coeff;
    for (size_t k = 1; k <= pow_x_ / 2; ++k) {
      coeff *=
          2.0 * (static_cast<double>(n) - static_cast<double>(k) + 1.0) /
          (2.0 * static_cast<double>(n) + 2.0 * static_cast<double>(k) + 3.0);
      result[2 * k + 1] = coeff * (4.0 * static_cast<double>(k) + 3.0);
    }
  }
  return result;
}

// See, for example, Equation (2.14) of Chebyshev Polynomials by Mason and
// Handscomb (2002)
template <>
DataVector Monomial::modes<Spectral::Basis::Chebyshev>() const {
  const size_t n = pow_x_;
  if (n == 0) {
    return DataVector{1.0};
  }
  DataVector result{n + 1, 0.0};
  const double c = 2.0 / static_cast<double>(two_to_the(n));
  for (size_t k = 0; k <= n / 2; ++k) {
    result[n - 2 * k] = c * static_cast<double>(binomial(n, k));
  }
  if (n % 2 == 0) {
    result[0] *= 0.5;
  }
  return result;
}
}  // namespace PolynomialTestFunctions
