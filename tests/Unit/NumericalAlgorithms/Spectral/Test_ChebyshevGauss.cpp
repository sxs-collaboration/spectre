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
                                   Spectral::Quadrature::Gauss>(num_points);
  CHECK_ITERABLE_APPROX(expected_points, points);
  // We test the \f$w_k\f$ here directly. Test_Spectral.cpp and
  // Test_DefiniteIntegral.cpp take care of testing the
  // `Spectral::quadrature_weights`.
  const auto weights =
      Spectral::compute_collocation_points_and_weights<
          Spectral::Basis::Chebyshev, Spectral::Quadrature::Gauss>(num_points)
          .second;
  CHECK_ITERABLE_APPROX(expected_weights, weights);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.ChebyshevGauss.PointsAndWeights",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  // Compare Chebyshev-Gauss points and weights to the analytic expressions
  // \f$\x_j=-\cos{\frac{2j+1}{2p+2}\pi}\f$ and \f$w_k=\frac{\pi}{N}\f$, where
  // \f$p=N-1\f$ is the polynomial degree, \f$N\f$ the number of collocation
  // points and \f$0\leq j \leq p\f$ (Kopriva, eqn. (1.128)).

  SECTION("Check 1 point") {
    test_points_and_weights(1, DataVector{0.}, DataVector{3.141592653589793});
  }

  SECTION("Check 2 points") {
    test_points_and_weights(2,
                            DataVector{-0.707106781186548, 0.7071067811865475},
                            DataVector(size_t{2}, 1.570796326794897));
  }
  SECTION("Check 3 points") {
    test_points_and_weights(
        3, DataVector{-0.866025403784439, 0., 0.866025403784439},
        DataVector(size_t{3}, 1.047197551196598));
  }
  SECTION("Check 4 points") {
    test_points_and_weights(4,
                            DataVector{-0.923879532511287, -0.382683432365090,
                                       0.3826834323650898, 0.9238795325112868},
                            DataVector(size_t{4}, 0.785398163397448));
  }
  SECTION("Check 5 points") {
    test_points_and_weights(
        5,
        DataVector{-0.951056516295154, -0.587785252292473, 0.,
                   0.5877852522924731, 0.9510565162951536},
        DataVector(size_t{5}, 0.628318530717959));
  }
  SECTION("Check 6 points") {
    test_points_and_weights(
        6,
        DataVector{-0.965925826289068, -0.707106781186548, -0.258819045102521,
                   0.258819045102521, 0.7071067811865475, 0.965925826289068},
        DataVector(size_t{6}, 0.523598775598299));
  }
}
