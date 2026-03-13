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
    if constexpr (basis == Basis::ZernikeB1) {
      // The integral of a constant is xi. Actual logical coordinates from
      // [0,1], with a orthogonality weighting of 1, thus 1.
      CHECK(sum(weights_n) == approx(1.0));
    } else if constexpr (basis == Basis::ZernikeB2) {
      // The integral of a constant is xi. Actual logical coordinates from
      // [0,1], with a orthogonality weighting of r, thus 1/2.
      CHECK(sum(weights_n) == approx(0.5));
    } else if constexpr (basis == Basis::ZernikeB3) {
      // The integral of a constant is xi. Actual logical coordinates from
      // [0,1], with a orthogonality weighting of r^2, thus 1/3.
      CHECK(sum(weights_n) == approx(1. / 3.));
    } else {
      // The integral of a constant is xi.  Logical coordintes from [-1, 1].
      // The definite integral is thus 2.
      CHECK(sum(weights_n) == approx(2.0));
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.QuadratureWeights",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  test<Basis::Legendre, Quadrature::Gauss>();
  test<Basis::Legendre, Quadrature::GaussLobatto>();
  test<Basis::Chebyshev, Quadrature::Gauss>();
  test<Basis::Chebyshev, Quadrature::GaussLobatto>();
  test<Basis::ZernikeB1, Quadrature::GaussRadauUpper>();
  test<Basis::ZernikeB2, Quadrature::GaussRadauUpper>();
  test<Basis::ZernikeB3, Quadrature::GaussRadauUpper>();
}
}  // namespace Spectral
