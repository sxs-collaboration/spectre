// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "Helpers/NumericalAlgorithms/Spectral/PolynomialTestFunctions.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"

namespace Spectral {
namespace {
DataVector expected_modes(const size_t pow_x) {
  switch (pow_x) {
    case 0:
      return DataVector{1.0};
    case 1:
      return DataVector{0.0, 1.0};
    case 2:
      return DataVector{1.0/3.0, 0.0, 2.0/3.0};
    case 3:
      return DataVector{0.0, 0.6, 0.0, 0.4};
    case 4:
      return DataVector{0.2, 0.0, 4.0/7.0, 0.0, 8.0/35.0};
    default:
      return DataVector{};
  }
}

void test_modes() {
  for (size_t pow_x = 0; pow_x < 5; ++pow_x) {
    const PolynomialTestFunctions::Monomial f{pow_x};
    CHECK_ITERABLE_APPROX(expected_modes(pow_x), f.modes<Basis::Legendre>());
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.BasisFunctions.Legendre",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  PolynomialTestFunctions::test_orthogonal_polynomial<
      Basis::Legendre, Quadrature::GaussLobatto>();
  PolynomialTestFunctions::test_orthogonal_polynomial<Basis::Legendre,
                                                      Quadrature::Gauss>();
  test_modes();
}
}  // namespace Spectral
