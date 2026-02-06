// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Helpers/DataStructures/ApplyMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/IntegrationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/MaximumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/MinimumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"

namespace Spectral {
namespace {
template <Basis basis, Quadrature quadrature>
void test() {
  CAPTURE(basis);
  CAPTURE(quadrature);
  for (size_t n = minimum_number_of_points<basis, quadrature>;
       n <= maximum_number_of_points<basis>; ++n) {
    CAPTURE(n);
    const Matrix& m = integration_matrix<basis, quadrature>(n);
    if (UNLIKELY(n == 1)) {
      // Cannot represent the indefinite integral of a constant with a
      // constant
      CHECK(m == Matrix{n, n, 0.0});
    } else {
      // The integral of one should be xi plus a constant. The indefinite
      // integral determines the integration constant by making the integral be
      // zero at xi = - 1.  Therefore the expected answer is xi + 1.
      const DataVector one{n, 1.0};
      const DataVector should_be_xi_plus_one = apply_matrix(m, one);
      const DataVector xi_plus_one =
          one + collocation_points<basis, quadrature>(n);
      CHECK_ITERABLE_APPROX(should_be_xi_plus_one, xi_plus_one);
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.IntegrationMatrix",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  test<Basis::Legendre, Quadrature::Gauss>();
  test<Basis::Legendre, Quadrature::GaussLobatto>();
  test<Basis::Chebyshev, Quadrature::Gauss>();
  test<Basis::Chebyshev, Quadrature::GaussLobatto>();
  // no known form of an integraion matrix for Zernike
  // there is no integration matrix for Fourier
}
}  // namespace Spectral
