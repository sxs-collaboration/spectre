// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/ApplyMatrix.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionValue.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/InterpolationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/MaximumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/MinimumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Gsl.hpp"

namespace Spectral {
namespace {
template <Basis basis, Quadrature quadrature>
void test(const DataVector& target_pts) {
  CAPTURE(basis);
  CAPTURE(quadrature);
  const auto custom_approx = basis == Basis::Fourier
                                 ? Approx::custom().epsilon(5.0e-12).scale(1.0)
                                 : Approx::custom().epsilon(5.0e-13).scale(1.0);
  for (size_t n = minimum_number_of_points<basis, quadrature>;
       n <= maximum_number_of_points<basis>; ++n) {
    CAPTURE(n);
    // If target points are source points, matrix should be identity
    const DataVector xi = collocation_points<basis, quadrature>(n);
    const Matrix should_be_identity =
        interpolation_matrix<basis, quadrature>(n, xi);
    Matrix identity{n, n, 0.0};
    for (size_t i = 0; i < n; ++i) {
      identity(i, i) = 1.0;
    }
    CHECK_ITERABLE_APPROX(should_be_identity, identity);
    const Matrix m = interpolation_matrix<basis, quadrature>(n, target_pts);
    REQUIRE(m.rows() == 5);
    REQUIRE(m.columns() == n);
    // Interpolating one should give one
    const DataVector one{n, 1.0};
    const DataVector should_be_one = apply_matrix(m, one);
    for (size_t i = 0; i < m.rows(); ++i) {
      CAPTURE(i);
      CHECK(should_be_one[i] == custom_approx(1.0));
    }
    // Interpolating a basis function
    for (size_t k = 0; k < n; ++k) {
      CAPTURE(k);
      const auto f_s = compute_basis_function_value<basis>(k, xi);
      const auto f_t = compute_basis_function_value<basis>(k, target_pts);
      const auto f_i = apply_matrix(m, f_s);
      CHECK_ITERABLE_CUSTOM_APPROX(f_i, f_t, custom_approx);
    }
  }
}
}  // namespace

// [[Timeout, 20]]
SPECTRE_TEST_CASE("Unit.Numerical.Spectral.InterpolationMatrix",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> xi_distribution(-1.0, 1.0);
  const auto xi_target_pts = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&xi_distribution), 5_st);
  test<Basis::Legendre, Quadrature::Gauss>(xi_target_pts);
  test<Basis::Legendre, Quadrature::GaussLobatto>(xi_target_pts);
  test<Basis::Chebyshev, Quadrature::Gauss>(xi_target_pts);
  test<Basis::Chebyshev, Quadrature::GaussLobatto>(xi_target_pts);
  std::uniform_real_distribution<> phi_distribution(0.0, 2.0 * M_PI);
  const auto phi_target_pts = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&phi_distribution), 5_st);
  test<Basis::Fourier, Quadrature::Equiangular>(phi_target_pts);
}
}  // namespace Spectral
