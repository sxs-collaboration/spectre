// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Helpers/DataStructures/ApplyMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/DifferentiationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/MaximumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/MinimumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Parity.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"

namespace Spectral {
namespace {
template <Basis basis, Quadrature quadrature>
void test() {
  CAPTURE(basis);
  CAPTURE(quadrature);
  const auto custom_approx = Approx::custom().epsilon(5.0e-13).scale(1.0);
  for (size_t n = minimum_number_of_points<basis, quadrature>;
       n <= maximum_number_of_points<basis>; ++n) {
    CAPTURE(n);
    const Matrix& m = differentiation_matrix<basis, quadrature>(n);
    const DataVector one{n, 1.0};
    const DataVector should_be_zero = apply_matrix(m, one);
    const DataVector zero{n, 0.0};
    CHECK_ITERABLE_CUSTOM_APPROX(should_be_zero, zero, custom_approx);
  }
}

template <Basis basis, Quadrature quadrature>
void test_with_parity() {
  CAPTURE(basis);
  CAPTURE(quadrature);
  const auto custom_approx = Approx::custom().epsilon(5.0e-13).scale(1.0);
  for (size_t n = minimum_number_of_points<basis, quadrature>;
       n <= maximum_number_of_points<basis>; ++n) {
    CAPTURE(n);
    const Matrix& m_even =
        differentiation_matrix<basis, quadrature>(n, Parity::Even);
    const DataVector one{n, 1.0};
    const DataVector should_be_zero = apply_matrix(m_even, one);
    const DataVector zero{n, 0.0};
    CHECK_ITERABLE_CUSTOM_APPROX(should_be_zero, zero, custom_approx);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.DifferentiationMatrix",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  test<Basis::Legendre, Quadrature::Gauss>();
  test<Basis::Legendre, Quadrature::GaussLobatto>();
  test<Basis::Chebyshev, Quadrature::Gauss>();
  test<Basis::Chebyshev, Quadrature::GaussLobatto>();
  test_with_parity<Basis::ZernikeB1, Quadrature::GaussRadauUpper>();
  test_with_parity<Basis::ZernikeB2, Quadrature::GaussRadauUpper>();
  test_with_parity<Basis::ZernikeB3, Quadrature::GaussRadauUpper>();
}
}  // namespace Spectral
