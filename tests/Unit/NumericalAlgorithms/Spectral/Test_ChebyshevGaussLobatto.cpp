// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

namespace {

void test_points_and_weights(const size_t num_points,
                             const DataVector expected_points,
                             const DataVector expected_weights) {
  const auto& points =
      Spectral::collocation_points<Spectral::Basis::Chebyshev,
                                   Spectral::Quadrature::GaussLobatto>(
          num_points);
  CHECK_ITERABLE_APPROX(expected_points, points);
  // We test the \f$w_k\f$ here directly. Test_Spectral.cpp and
  // Test_DefiniteIntegral.cpp take care of testing the
  // Spectral::quadrature_weights.
  const auto weights =
      Spectral::compute_collocation_points_and_weights<
          Spectral::Basis::Chebyshev, Spectral::Quadrature::GaussLobatto>(
          num_points)
          .second;
  CHECK_ITERABLE_APPROX(expected_weights, weights);
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Numerical.Spectral.ChebyshevGaussLobatto.PointsAndWeights",
    "[NumericalAlgorithms][Spectral][Unit]") {
  // Compare Chebyshev-Gauss-Lobatto points and weights to the analytic
  // expressions \f$\x_j=-\cos{\frac{j\pi}{p}}\f$ and \f$w_k=\frac{\pi}{p}\f$
  // except for \f$w_{0,p}=\frac{\pi}{2p}\f$, where \f$p=N-1\f$ is the
  // polynomial degree, \f$N\f$ the number of collocation points and \f$0\leq j
  // \leq p\f$ (Kopriva, eqn. (1.130)).

  SECTION("Check 2 points") {
    test_points_and_weights(2, DataVector{-1., 1.},
                            DataVector(size_t{2}, 1.570796326794897));
  }
  SECTION("Check 3 points") {
    test_points_and_weights(
        3, DataVector{-1., 0., 1.},
        DataVector{0.785398163397448, 1.570796326794897, 0.785398163397448});
  }
  SECTION("Check 4 points") {
    test_points_and_weights(4, DataVector{-1., -0.5, 0.5, 1.},
                            DataVector{0.523598775598299, 1.047197551196598,
                                       1.047197551196598, 0.523598775598299});
  }
  SECTION("Check 5 points") {
    test_points_and_weights(
        5, DataVector{-1., -0.707106781186548, 0., 0.7071067811865475, 1.},
        DataVector{0.3926990816987242, 0.785398163397448, 0.785398163397448,
                   0.785398163397448, 0.3926990816987242});
  }
  SECTION("Check 6 points") {
    test_points_and_weights(
        6,
        DataVector{-1., -0.809016994374947, -0.309016994374947,
                   0.3090169943749474, 0.809016994374947, 1.},
        DataVector{0.3141592653589793, 0.628318530717959, 0.628318530717959,
                   0.628318530717959, 0.628318530717959, 0.3141592653589793});
  }
}
