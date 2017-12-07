// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <catch.hpp>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/Spectral/LegendreGaussLobatto.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/Blas.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
SPECTRE_TEST_CASE("Unit.Numerical.Spectral.LegendreGaussLobatto.Points",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  // Compare LGL points to matlab code accompanying the
  // book "Nodal Discontinuous Galerkin Methods" by Hesthaven and Warburton
  // http://www.nudg.org/
  std::array<double, 1> collocation_pts_4{{.4472135954999579}};
  std::array<double, 1> collocation_pts_5{{.6546536707079773}};
  std::array<double, 2> collocation_pts_6{
      {.2852315164806452, .7650553239294646}};
  std::array<double, 2> collocation_pts_7{
      {.4688487934707142, .8302238962785669}};
  std::array<double, 3> collocation_pts_8{
      {.2092992179024791, .5917001814331421, .8717401485096066}};
  std::array<double, 3> collocation_pts_9{
      {.3631174638261783, .6771862795107377, .8997579954114600}};
  std::array<double, 4> collocation_pts_10{
      {.1652789576663869, .4779249498104444, .7387738651055048,
       .9195339081664589}};
  std::array<double, 9> collocation_pts_20{
      {.08054593723882171, .2395517059229869, .3923531837139090,
       .5349928640318858, .6637764022903113, .7753682609520557,
       .8668779780899503, .9359344988126655, .9807437048939139}};

  // cppcheck-suppress preprocessorErrorDirective
  SECTION("Check 2 points") {
    const DataVector& collocation_pts = Basis::lgl::collocation_points(2);
    CHECK(collocation_pts[0] == -1.0);
    CHECK(collocation_pts[1] == 1.0);
  }
  SECTION("Check 3 points") {
    const DataVector& collocation_pts = Basis::lgl::collocation_points(3);
    CHECK(collocation_pts[0] == -1.0);
    CHECK(collocation_pts[1] == approx(0));
    CHECK(collocation_pts[2] == 1.0);
  }
  SECTION("Check 4 points") {
    const DataVector& collocation_pts = Basis::lgl::collocation_points(4);
    CHECK(collocation_pts[0] == -1.0);
    CHECK(collocation_pts[1] == approx(-collocation_pts_4[0]));
    CHECK(collocation_pts[2] == approx(collocation_pts_4[0]));
    CHECK(collocation_pts[3] == 1.0);
  }
  SECTION("Check 5 points") {
    const DataVector& collocation_pts = Basis::lgl::collocation_points(5);
    CHECK(collocation_pts[0] == -1.0);
    CHECK(collocation_pts[1] == approx(-collocation_pts_5[0]));
    CHECK(collocation_pts[2] == approx(0.0));
    CHECK(collocation_pts[3] == approx(collocation_pts_5[0]));
    CHECK(collocation_pts[4] == 1.0);
  }
  SECTION("Check 6 points") {
    const DataVector& collocation_pts = Basis::lgl::collocation_points(6);
    CHECK(collocation_pts[0] == -1.0);
    CHECK(collocation_pts[1] == approx(-collocation_pts_6[1]));
    CHECK(collocation_pts[2] == approx(-collocation_pts_6[0]));
    CHECK(collocation_pts[3] == approx(collocation_pts_6[0]));
    CHECK(collocation_pts[4] == approx(collocation_pts_6[1]));
    CHECK(collocation_pts[5] == 1.0);
  }
  SECTION("Check 7 points") {
    const DataVector& collocation_pts = Basis::lgl::collocation_points(7);
    CHECK(collocation_pts[0] == -1.0);
    CHECK(collocation_pts[1] == approx(-collocation_pts_7[1]));
    CHECK(collocation_pts[2] == approx(-collocation_pts_7[0]));
    CHECK(collocation_pts[3] == approx(0.0));
    CHECK(collocation_pts[4] == approx(collocation_pts_7[0]));
    CHECK(collocation_pts[5] == approx(collocation_pts_7[1]));
    CHECK(collocation_pts[6] == 1.0);

    const DataVector& quadrature_weights = Basis::lgl::quadrature_weights(7);
    CHECK(quadrature_weights[0] == approx(4.761904761904762e-2));
    CHECK(quadrature_weights[1] == approx(0.276826047361566));
    CHECK(quadrature_weights[2] == approx(0.431745381209863));
    CHECK(quadrature_weights[3] == approx(0.487619047619048));
  }
  SECTION("Check 8 points") {
    const DataVector& collocation_pts = Basis::lgl::collocation_points(8);
    CHECK(collocation_pts[0] == -1.0);
    CHECK(collocation_pts[1] == approx(-collocation_pts_8[2]));
    CHECK(collocation_pts[2] == approx(-collocation_pts_8[1]));
    CHECK(collocation_pts[3] == approx(-collocation_pts_8[0]));
    CHECK(collocation_pts[4] == approx(collocation_pts_8[0]));
    CHECK(collocation_pts[5] == approx(collocation_pts_8[1]));
    CHECK(collocation_pts[6] == approx(collocation_pts_8[2]));
    CHECK(collocation_pts[7] == 1.0);
  }
  SECTION("Check 9 points") {
    const DataVector& collocation_pts = Basis::lgl::collocation_points(9);
    CHECK(collocation_pts[0] == -1.0);
    CHECK(collocation_pts[1] == approx(-collocation_pts_9[2]));
    CHECK(collocation_pts[2] == approx(-collocation_pts_9[1]));
    CHECK(collocation_pts[3] == approx(-collocation_pts_9[0]));
    CHECK(collocation_pts[4] == approx(0.0));
    CHECK(collocation_pts[5] == approx(collocation_pts_9[0]));
    CHECK(collocation_pts[6] == approx(collocation_pts_9[1]));
    CHECK(collocation_pts[7] == approx(collocation_pts_9[2]));
    CHECK(collocation_pts[8] == 1.0);
  }
  SECTION("Check 10 points") {
    const DataVector& collocation_pts = Basis::lgl::collocation_points(10);
    CHECK(collocation_pts[0] == -1.0);
    CHECK(collocation_pts[1] == approx(-collocation_pts_10[3]));
    CHECK(collocation_pts[2] == approx(-collocation_pts_10[2]));
    CHECK(collocation_pts[3] == approx(-collocation_pts_10[1]));
    CHECK(collocation_pts[4] == approx(-collocation_pts_10[0]));
    CHECK(collocation_pts[5] == approx(collocation_pts_10[0]));
    CHECK(collocation_pts[6] == approx(collocation_pts_10[1]));
    CHECK(collocation_pts[7] == approx(collocation_pts_10[2]));
    CHECK(collocation_pts[8] == approx(collocation_pts_10[3]));
    CHECK(collocation_pts[9] == 1.0);
  }
  if (Basis::lgl::maximum_number_of_pts > 19) {
    SECTION("Check 20 points") {
      const DataVector& collocation_pts = Basis::lgl::collocation_points(20);
      CHECK(collocation_pts[0] == -1.0);
      CHECK(collocation_pts[1] == approx(-collocation_pts_20[8]));
      CHECK(collocation_pts[2] == approx(-collocation_pts_20[7]));
      CHECK(collocation_pts[3] == approx(-collocation_pts_20[6]));
      CHECK(collocation_pts[4] == approx(-collocation_pts_20[5]));
      CHECK(collocation_pts[5] == approx(-collocation_pts_20[4]));
      CHECK(collocation_pts[6] == approx(-collocation_pts_20[3]));
      CHECK(collocation_pts[7] == approx(-collocation_pts_20[2]));
      CHECK(collocation_pts[8] == approx(-collocation_pts_20[1]));
      CHECK(collocation_pts[9] == approx(-collocation_pts_20[0]));
      CHECK(collocation_pts[10] == approx(collocation_pts_20[0]));
      CHECK(collocation_pts[11] == approx(collocation_pts_20[1]));
      CHECK(collocation_pts[12] == approx(collocation_pts_20[2]));
      CHECK(collocation_pts[13] == approx(collocation_pts_20[3]));
      CHECK(collocation_pts[14] == approx(collocation_pts_20[4]));
      CHECK(collocation_pts[15] == approx(collocation_pts_20[5]));
      CHECK(collocation_pts[16] == approx(collocation_pts_20[6]));
      CHECK(collocation_pts[17] == approx(collocation_pts_20[7]));
      CHECK(collocation_pts[18] == approx(collocation_pts_20[8]));
      CHECK(collocation_pts[19] == 1.0);
    }
  }
}

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.LegendreGaussLobatto.DiffMatrix",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  // Compare differentiation matrix values to matlab code accompanying the
  // book "Nodal Discontinuous Galerkin Methods" by Hesthaven and Warburton
  // http://www.nudg.org/
  //
  // Command to generate differentation matrix with the Matlab code:
  // DN = Dmatrix1D(N, JacobiGL(0,0,N), Vandermonde1D(N, JacobiGL(0,0,N)))

  SECTION("Check 2 points") {
    const Matrix diff_matrix_expected = []() {
      Matrix diff_matrix(2, 2);
      diff_matrix(0, 0) = -.5;
      diff_matrix(0, 1) = .5;
      diff_matrix(1, 0) = -.5;
      diff_matrix(1, 1) = .5;
      return diff_matrix;
    }();
    const Matrix& diff_matrix = Basis::lgl::differentiation_matrix(2);
    for (size_t i = 0; i < 2; ++i) {
      for (size_t j = 0; j < 2; ++j) {
        CHECK(diff_matrix(i, j) == approx(diff_matrix_expected(i, j)));
      }
    }
  }

  SECTION("Check 3 points") {
    const Matrix diff_matrix_expected = []() {
      Matrix diff_matrix(3, 3);
      diff_matrix(0, 0) = -1.5;
      diff_matrix(0, 1) = 2.0;
      diff_matrix(0, 2) = -.5;
      diff_matrix(1, 0) = -.5;
      diff_matrix(1, 1) = 0.0;
      diff_matrix(1, 2) = .5;
      diff_matrix(2, 0) = 0.5;
      diff_matrix(2, 1) = -2;
      diff_matrix(2, 2) = 1.5;
      return diff_matrix;
    }();
    const Matrix& diff_matrix = Basis::lgl::differentiation_matrix(3);
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        CHECK(diff_matrix(i, j) == approx(diff_matrix_expected(i, j)));
      }
    }
  }

  SECTION("Check 4 points") {
    const Matrix diff_matrix_expected = []() {
      Matrix diff_matrix(4, 4);
      diff_matrix(0, 0) = -3;
      diff_matrix(0, 1) = 4.045084971874735;
      diff_matrix(0, 2) = -1.545084971874736;
      diff_matrix(0, 3) = .5;
      diff_matrix(1, 0) = -.8090169943749473;
      diff_matrix(1, 1) = 0;
      diff_matrix(1, 2) = 1.118033988749895;
      diff_matrix(1, 3) = -.3090169943749473;
      diff_matrix(2, 0) = .3090169943749472;
      diff_matrix(2, 1) = -1.118033988749895;
      diff_matrix(2, 2) = 0;
      diff_matrix(2, 3) = .8090169943749475;
      diff_matrix(3, 0) = -.5;
      diff_matrix(3, 1) = 1.545084971874735;
      diff_matrix(3, 2) = -4.045084971874735;
      diff_matrix(3, 3) = 3.0;
      return diff_matrix;
    }();
    const Matrix& diff_matrix = Basis::lgl::differentiation_matrix(4);
    for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        CHECK(diff_matrix(i, j) == approx(diff_matrix_expected(i, j)));
      }
    }
  }

  SECTION("Check 5 points") {
    const Matrix diff_matrix_expected = []() {
      Matrix diff_matrix(5, 5);
      diff_matrix(0, 0) = -5;
      diff_matrix(0, 1) = 6.756502488724239;
      diff_matrix(0, 2) = -2.666666666666666;
      diff_matrix(0, 3) = 1.410164177942428;
      diff_matrix(0, 4) = -.5;
      diff_matrix(1, 0) = -1.240990253030983;
      diff_matrix(1, 1) = 0;
      diff_matrix(1, 2) = 1.745743121887939;
      diff_matrix(1, 3) = -0.763762615825973;
      diff_matrix(1, 4) = 0.259009746969017;
      diff_matrix(2, 0) = 0.375000000000000;
      diff_matrix(2, 1) = -1.336584577695454;
      diff_matrix(2, 2) = 0;
      diff_matrix(2, 3) = 1.336584577695453;
      diff_matrix(2, 4) = -0.375000000000000;
      diff_matrix(3, 0) = -0.259009746969017;
      diff_matrix(3, 1) = 0.763762615825973;
      diff_matrix(3, 2) = -1.745743121887939;
      diff_matrix(3, 3) = 0;
      diff_matrix(3, 4) = 1.240990253030984;
      diff_matrix(4, 0) = .5;
      diff_matrix(4, 1) = -1.410164177942428;
      diff_matrix(4, 2) = 2.666666666666666;
      diff_matrix(4, 3) = -6.756502488724238;
      diff_matrix(4, 4) = 5;
      return diff_matrix;
    }();
    const Matrix& diff_matrix = Basis::lgl::differentiation_matrix(5);
    for (size_t i = 0; i < 5; ++i) {
      for (size_t j = 0; j < 5; ++j) {
        CHECK(diff_matrix(i, j) == approx(diff_matrix_expected(i, j)));
      }
    }
  }

  SECTION("Check 6 points") {
    const Matrix diff_matrix_expected = []() {
      Matrix diff_matrix(6, 6);
      diff_matrix(0, 0) = -7.5;
      diff_matrix(0, 1) = 10.141415936319671;
      diff_matrix(0, 2) = -4.036187270305348;
      diff_matrix(0, 3) = 2.244684648176167;
      diff_matrix(0, 4) = -1.349913314190486;
      diff_matrix(0, 5) = .5;
      diff_matrix(1, 0) = -1.786364948339096;
      diff_matrix(1, 1) = 0;
      diff_matrix(1, 2) = 2.523426777429454;
      diff_matrix(1, 3) = -1.152828158535928;
      diff_matrix(1, 4) = 0.653547507429800;
      diff_matrix(1, 5) = -0.237781177984231;
      diff_matrix(2, 0) = 0.484951047853570;
      diff_matrix(2, 1) = -1.721256952830234;
      diff_matrix(2, 2) = 0;
      diff_matrix(2, 3) = 1.752961966367866;
      diff_matrix(2, 4) = -0.786356672223241;
      diff_matrix(2, 5) = 0.269700610832039;
      diff_matrix(3, 0) = -0.269700610832039;
      diff_matrix(3, 1) = 0.786356672223241;
      diff_matrix(3, 2) = -1.752961966367866;
      diff_matrix(3, 3) = 0;
      diff_matrix(3, 4) = 1.721256952830234;
      diff_matrix(3, 5) = -0.484951047853569;
      diff_matrix(4, 0) = 0.237781177984231;
      diff_matrix(4, 1) = -0.653547507429801;
      diff_matrix(4, 2) = 1.152828158535929;
      diff_matrix(4, 3) = -2.523426777429456;
      diff_matrix(4, 4) = 0;
      diff_matrix(4, 5) = 1.786364948339093;
      diff_matrix(5, 0) = -.5;
      diff_matrix(5, 1) = 1.349913314190487;
      diff_matrix(5, 2) = -2.244684648176166;
      diff_matrix(5, 3) = 4.036187270305350;
      diff_matrix(5, 4) = -10.141415936319667;
      diff_matrix(5, 5) = 7.5;
      return diff_matrix;
    }();
    const Matrix& diff_matrix = Basis::lgl::differentiation_matrix(6);
    for (size_t i = 0; i < 6; ++i) {
      for (size_t j = 0; j < 6; ++j) {
        CHECK(diff_matrix(i, j) == approx(diff_matrix_expected(i, j)));
      }
    }
  }
}

SPECTRE_TEST_CASE(
    "Unit.Numerical.Spectral.LegendreGaussLobatto.LinearFilterMatrix",
    "[NumericalAlgorithms][Spectral][Unit]") {
  for (size_t n = 2; n < 10; ++n) {
    const Matrix& filter_matrix = Basis::lgl::linear_filter_matrix(n);
    const Matrix& grid_points_to_spectral_matrix =
        Basis::lgl::grid_points_to_spectral_matrix(n);
    const DataVector& collocation_points = Basis::lgl::collocation_points(n);
    DataVector u(n);
    for (size_t s = 0; s < n; ++s) {
      u[s] = exp(collocation_points[s]);
    }
    DataVector u_filtered(n);
    dgemv_('N', n, n, 1.0, filter_matrix.data(), n, u.data(), 1, 0.0,
           u_filtered.data(), 1);
    DataVector u_spectral(n);
    dgemv_('N', n, n, 1.0, grid_points_to_spectral_matrix.data(), n,
           u_filtered.data(), 1, 0.0, u_spectral.data(), 1);
    for (size_t s = 2; s < n; ++s) {
      CHECK(0.0 == approx(u_spectral[s]));
    }
  }
}

SPECTRE_TEST_CASE(
    "Unit.Numerical.Spectral.LegendreGaussLobatto.InterpolationMatrix",
    "[NumericalAlgorithms][Spectral][Unit]") {
  auto check_interp = [](const size_t num_pts, auto func) {
    const DataVector& collocation_points =
        Basis::lgl::collocation_points(num_pts);
    DataVector u(num_pts);
    for (size_t i = 0; i < num_pts; ++i) {
      u[i] = func(collocation_points[i]);
    }
    DataVector new_points{-0.5, -0.4837, 0.5, 0.9378, 1.0};
    DataVector interpolated_u(new_points.size(), 0.0);

    const Matrix interp_matrix =
        Basis::lgl::interpolation_matrix(num_pts, new_points);
    dgemv_('n', new_points.size(), num_pts, 1.0, interp_matrix.data(),
           new_points.size(), u.data(), 1, 0.0, interpolated_u.data(), 1);

    CHECK(interpolated_u.size() == new_points.size());
    for (size_t i = 0; i < new_points.size(); ++i) {
      CHECK(func(new_points[i]) == approx(interpolated_u[i]));
    }
  };

  check_interp(2, [](const double x) { return x + 1.0; });
  check_interp(3, [](const double x) { return x * x + x + 1.0; });
  check_interp(4,
               [](const double x) { return pow<3>(x) + pow<2>(x) + x + 1.0; });
  check_interp(5, [](const double x) {
    return pow<4>(x) + pow<3>(x) + pow<2>(x) + x + 1.0;
  });
  check_interp(6, [](const double x) {
    return pow<5>(x) + pow<4>(x) + pow<3>(x) + pow<2>(x) + x + 1.0;
  });
  check_interp(7, [](const double x) {
    return pow<6>(x) + pow<5>(x) + pow<4>(x) + pow<3>(x) + pow<2>(x) + x + 1.0;
  });
  check_interp(8, [](const double x) {
    return pow<7>(x) + pow<6>(x) + pow<5>(x) + pow<4>(x) + pow<3>(x) +
           pow<2>(x) + x + 1.0;
  });
  check_interp(9, [](const double x) {
    return pow<8>(x) + pow<7>(x) + pow<6>(x) + pow<5>(x) + pow<4>(x) +
           pow<3>(x) + pow<2>(x) + x + 1.0;
  });
}
}  // namespace
