// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionNormalizationSquare.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionValue.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPointsAndWeights.hpp"
#include "NumericalAlgorithms/Spectral/MaximumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/MinimumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace Spectral {
namespace {
template <Basis basis, Quadrature quadrature>
void test() {
  CAPTURE(basis);
  CAPTURE(quadrature);
  for (size_t n = minimum_number_of_points<basis, quadrature>;
       n <= maximum_number_of_points<basis>; ++n) {
    CAPTURE(n);
    const auto& [xi, w] =
        compute_collocation_points_and_weights<basis, quadrature>(n);
    const size_t k_max =
        quadrature == Quadrature::Gauss ? n - 1 : (n < 2 ? 0 : n - 2);
    CAPTURE(k_max);
    for (size_t k = 0; k <= k_max; ++k) {
      CAPTURE(k);
      const auto f = compute_basis_function_value<basis>(k, xi);
      const double expected = sum(square(f) * w);
      CHECK(approx(expected) ==
            compute_basis_function_normalization_square<basis>(k));
    }
  }
}

template <Basis basis, Quadrature quadrature>
void test_two_index() {
  static_assert(basis == Basis::ZernikeB1 or basis == Basis::ZernikeB2 or
                basis == Basis::ZernikeB3);
  const auto custom_approx = Approx::custom().epsilon(1e-12);
  CAPTURE(basis);
  CAPTURE(quadrature);
  for (size_t n = minimum_number_of_points<basis, quadrature>;
       n <= maximum_number_of_points<basis>; ++n) {
    CAPTURE(n);
    const auto& [xi, w] =
        compute_collocation_points_and_weights<basis, quadrature>(n);
    const size_t k_max = n < 2 ? 0 : n - 2;
    for (size_t k = 0; k <= k_max; ++k) {
      CAPTURE(k);
      for (size_t m = k % 2; m <= k; m += 2) {
        CAPTURE(m);
        const auto f = compute_basis_function_value<basis>(k, m, xi);
        const double expected = sum(square(f) * w);
        CHECK_ITERABLE_CUSTOM_APPROX(
            expected, compute_basis_function_normalization_square<basis>(k),
            custom_approx);
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.BasisFunctionNormalizationSquare",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  test<Basis::Legendre, Quadrature::Gauss>();
  test<Basis::Legendre, Quadrature::GaussLobatto>();
  test<Basis::Chebyshev, Quadrature::Gauss>();
  test<Basis::Chebyshev, Quadrature::GaussLobatto>();
  test<Basis::Fourier, Quadrature::Equiangular>();
  test_two_index<Basis::ZernikeB1, Quadrature::GaussRadauUpper>();
  test_two_index<Basis::ZernikeB2, Quadrature::GaussRadauUpper>();
  test_two_index<Basis::ZernikeB3, Quadrature::GaussRadauUpper>();
}
}  // namespace Spectral
