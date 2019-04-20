// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

namespace {

void test_points_and_weights(const size_t num_points,
                             const DataVector expected_points,
                             const DataVector expected_weights) {
  const auto& points =
      Spectral::collocation_points<Spectral::Basis::Legendre,
                                   Spectral::Quadrature::Gauss>(num_points);
  CHECK_ITERABLE_APPROX(expected_points, points);
  const auto& weights =
      Spectral::quadrature_weights<Spectral::Basis::Legendre,
                                   Spectral::Quadrature::Gauss>(num_points);
  CHECK_ITERABLE_APPROX(expected_weights, weights);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.LegendreGauss.PointsAndWeights",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  // Compare LG points to matlab code accompanying the
  // book "Nodal Discontinuous Galerkin Methods" by Hesthaven and Warburton
  // available at http://www.nudg.org/. To compute `num_points` LG points, run
  // the routine `JacobiGQ(0, 0, num_points - 1)`.
  // Quadrature weights computed in Mathematica, code available on request from
  // nils.fischer@aei.mpg.de.

  SECTION("Check 1 point") {
    test_points_and_weights(1, DataVector{0.}, DataVector{2.});
  }

  SECTION("Check 2 points") {
    test_points_and_weights(2,
                            DataVector{-0.577350269189626, 0.577350269189626},
                            DataVector{1., 1.});
  }
  SECTION("Check 3 points") {
    test_points_and_weights(
        3, DataVector{-0.774596669241483, 0., 0.774596669241483},
        DataVector{{0.555555555555556, 0.888888888888888, 0.555555555555556}});
  }
  SECTION("Check 4 points") {
    test_points_and_weights(4,
                            DataVector{-0.861136311594053, -0.339981043584856,
                                       0.339981043584856, 0.861136311594053},
                            DataVector{{0.347854845137454, 0.652145154862546,
                                        0.652145154862546, 0.347854845137454}});
  }
  SECTION("Check 5 points") {
    test_points_and_weights(
        5,
        DataVector{-0.906179845938664, -0.538469310105683, 0.,
                   0.538469310105683, 0.906179845938664},
        DataVector{{0.236926885056189, 0.478628670499366, 0.568888888888889,
                    0.478628670499366, 0.236926885056189}});
  }
}

namespace {

void test_diff_matrix(const size_t num_points, const Matrix& expected_matrix) {
  const auto& diff_matrix =
      Spectral::differentiation_matrix<Spectral::Basis::Legendre,
                                       Spectral::Quadrature::Gauss>(num_points);
  CHECK_MATRIX_APPROX(expected_matrix, diff_matrix);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.LegendreGauss.DiffMatrix",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  // Compare differentiation matrix values to matlab code accompanying the
  // book "Nodal Discontinuous Galerkin Methods" by Hesthaven and Warburton
  // http://www.nudg.org/
  //
  // Command to generate differentation matrix with the Matlab code:
  // DN = Dmatrix1D(N, JacobiGQ(0,0,N), Vandermonde1D(N, JacobiGQ(0,0,N)))

  SECTION("Check 1 point") { test_diff_matrix(1, Matrix(1, 1, 0.)); }

  SECTION("Check 2 points") {
    test_diff_matrix(2, Matrix({{-0.866025403784438, 0.866025403784438},
                                {-0.866025403784438, 0.866025403784438}}));
  }

  SECTION("Check 3 points") {
    test_diff_matrix(
        3, Matrix({{-1.93649167310371, 2.58198889747161, -0.645497224367903},
                   {-0.645497224367903, 0, 0.645497224367903},
                   {0.645497224367903, -2.58198889747161, 1.93649167310371}}));
  }

  SECTION("Check 4 points") {
    test_diff_matrix(4, Matrix({{-3.33200023635228, 4.86015441568520,
                                 -2.10878234849518, 0.580628169162264},
                                {-0.757557614799232, -0.384414392223212,
                                 1.47067023128072, -0.328698224258274},
                                {0.328698224258274, -1.47067023128072,
                                 0.384414392223213, 0.757557614799232},
                                {-0.580628169162264, 2.10878234849518,
                                 -4.86015441568520, 3.33200023635228}}));
  }

  SECTION("Check 5 points") {
    test_diff_matrix(
        5, Matrix({{-5.06704059565454, 7.70195208517225, -4.04354375438766,
                    1.96039911583328, -0.551766850963317},
                   {-0.960256023631957, -0.758353217167877, 2.40275065216430,
                    -0.928558026643834, 0.244416615279368},
                   {0.301168159727831, -1.43538824233474, 0, 1.43538824233474,
                    -0.301168159727831},
                   {-0.244416615279368, 0.928558026643835, -2.40275065216430,
                    0.758353217167876, 0.960256023631957},
                   {0.551766850963317, -1.96039911583328, 4.04354375438767,
                    -7.70195208517225, 5.06704059565454}}));
  }
}

namespace {

void test_modal_to_nodal_matrix(const size_t num_points,
                                const Matrix& expected_matrix) {
  const auto& matrix =
      Spectral::modal_to_nodal_matrix<Spectral::Basis::Legendre,
                                      Spectral::Quadrature::Gauss>(num_points);
  CHECK_MATRIX_APPROX(expected_matrix, matrix);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.LegendreGauss.ModalToNodal",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  SECTION("Check 1 point") { test_modal_to_nodal_matrix(1, Matrix(1, 1, 1.)); }
  SECTION("Check 2 points") {
    test_modal_to_nodal_matrix(
        2, Matrix({{1., -0.5773502691896258}, {1., 0.5773502691896258}}));
  }
  SECTION("Check 3 points") {
    test_modal_to_nodal_matrix(3, Matrix({{1., -0.7745966692414830, 0.4},
                                          {1., 0., -0.5},
                                          {1., 0.7745966692414830, 0.4}}));
  }
  SECTION("Check 4 points") {
    test_modal_to_nodal_matrix(
        4,
        Matrix(
            {{1., -0.8611363115940530, 0.6123336207187148, -0.3047469849552077},
             {1., -0.3399810435848560, -0.3266193350044284, 0.4117279996728994},
             {1., 0.3399810435848560, -0.3266193350044284, -0.4117279996728994},
             {1., 0.8611363115940530, 0.6123336207187148,
              0.3047469849552077}}));
  }
  SECTION("Check 5 points") {
    test_modal_to_nodal_matrix(
        5, Matrix({{1., -0.9061798459386637, 0.7317428697781310,
                    -0.5010311710446620, 0.2457354590949121},
                   {1., -0.5384693101056831, -0.06507620311146464,
                    0.4173821037266682, -0.3445008911936774},
                   {1., 0., -0.4999999999999999, 0., 0.3749999999999999},
                   {1., 0.5384693101056831, -0.06507620311146464,
                    -0.4173821037266682, -0.3445008911936774},
                   {1., 0.9061798459386637, 0.7317428697781310,
                    0.5010311710446620, 0.2457354590949121}}));
  }
}

namespace {

void test_nodal_to_modal_matrix(const size_t num_points,
                                const Matrix& expected_matrix) {
  const auto& matrix =
      Spectral::nodal_to_modal_matrix<Spectral::Basis::Legendre,
                                      Spectral::Quadrature::Gauss>(num_points);
  CHECK_MATRIX_APPROX(expected_matrix, matrix);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.LegendreGauss.NodalToModal",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  SECTION("Check 1 point") { test_nodal_to_modal_matrix(1, Matrix(1, 1, 1.)); }
  SECTION("Check 2 points") {
    test_nodal_to_modal_matrix(
        2, Matrix({{0.5, 0.5}, {-0.8660254037844385, 0.8660254037844385}}));
  }
  SECTION("Check 3 points") {
    test_nodal_to_modal_matrix(
        3,
        Matrix({{0.2777777777777781, 0.4444444444444438, 0.2777777777777781},
                {-0.6454972243679031, 0, 0.6454972243679031},
                {0.5555555555555561, -1.111111111111112, 0.5555555555555561}}));
  }
  SECTION("Check 4 points") {
    test_nodal_to_modal_matrix(
        4, Matrix({{0.1739274225687268, 0.3260725774312732, 0.3260725774312732,
                    0.1739274225687269},
                   {-0.4493256574676805, -0.3325754854784655,
                    0.3325754854784655, 0.4493256574676804},
                   {0.5325080420189108, -0.5325080420189108,
                    -0.5325080420189108, 0.5325080420189108},
                   {-0.3710270034019465, 0.9397724703777526,
                    -0.9397724703777526, 0.3710270034019466}}));
  }
  SECTION("Check 5 points") {
    test_nodal_to_modal_matrix(
        5,
        Matrix({{0.1184634425280945, 0.2393143352496832, 0.2844444444444443,
                 0.2393143352496832, 0.1184634425280947},
                {-0.3220475522984176, -0.3865902750008912, 0.,
                 0.3865902750008913, 0.3220475522984176},
                {0.4334238969965231, -0.07786834144096745, -0.7111111111111112,
                 -0.07786834144096744, 0.4334238969965231},
                {-0.4154771413508326, 0.6991986448892331, 0.,
                 -0.6991986448892332, 0.4154771413508326},
                {0.2619960159204452, -0.7419960159204453, 0.96,
                 -0.7419960159204452, 0.2619960159204453}}));
  }
}
