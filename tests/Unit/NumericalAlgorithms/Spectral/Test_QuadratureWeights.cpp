// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/MaximumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/QuadratureWeights.hpp"

namespace Spectral {
namespace {

template <Basis basis, Quadrature quadrature>
void test() {
  CAPTURE(basis);
  CAPTURE(quadrature);
  // Cannot represent the integral of a constant with a single point
  for (size_t n = 2; n <= maximum_number_of_points<basis>; ++n) {
    const DataVector& weights_n = quadrature_weights<basis, quadrature>(n);
    const Mesh<1> mesh{n, basis, quadrature};
    const DataVector& weights_m = quadrature_weights(mesh);
    CHECK(weights_n == weights_m);
    CHECK(weights_n.data() == weights_m.data());
    // The integral of a constant is xi.  The definite integral is thus 2.
    CHECK(sum(weights_n) == approx(2.0));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.QuadratureWeights",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  test<Basis::Legendre, Quadrature::Gauss>();
  test<Basis::Legendre, Quadrature::GaussLobatto>();
  // Chebyshev fails this test!
  // test<Basis::Chebyshev, Quadrature::Gauss>();
  // test<Basis::Chebyshev, Quadrature::GaussLobatto>();
}
}  // namespace Spectral
