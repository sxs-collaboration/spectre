// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <numbers>

#include "DataStructures/DataVector.hpp"
#include "Helpers/NumericalAlgorithms/Spectral/BallTestFunctions.hpp"

namespace {

// Verify each derivative against a central finite difference at one point.
// df_dph is the Pfaffian derivative: (1/sin \theta) * \partial f/\partial \phi.
void check_derivatives(const BallTestFunctions::ProductOfPolynomials& f,
                       const double r, const double theta, const double phi) {
  const double eps = 1.0e-6;
  const DataVector r_v{r};
  const DataVector th_v{theta};
  const DataVector ph_v{phi};

  const double df_dr_fd =
      (f(r + eps, theta, phi) - f(r - eps, theta, phi)) / (2.0 * eps);
  CHECK(f.df_dr(r_v, th_v, ph_v)[0] == approx(df_dr_fd).epsilon(1.0e-5));

  const double df_dth_fd =
      (f(r, theta + eps, phi) - f(r, theta - eps, phi)) / (2.0 * eps);
  CHECK(f.df_dth(r_v, th_v, ph_v)[0] == approx(df_dth_fd).epsilon(1.0e-5));

  const double df_dph_fd = (f(r, theta, phi + eps) - f(r, theta, phi - eps)) /
                           (2.0 * eps * sin(theta));
  CHECK(f.df_dph(r_v, th_v, ph_v)[0] == approx(df_dph_fd).epsilon(1.0e-5));
}

void test_function_values() {
  const double r = 0.7;
  const double theta = 0.9;
  const double phi = 1.2;
  const double x = r * sin(theta) * cos(phi);
  const double y = r * sin(theta) * sin(phi);
  const double z = r * cos(theta);

  CHECK(BallTestFunctions::ProductOfPolynomials{0, 0, 0}(r, theta, phi) ==
        approx(1.0));
  CHECK(BallTestFunctions::ProductOfPolynomials{1, 0, 0}(r, theta, phi) ==
        approx(x));
  CHECK(BallTestFunctions::ProductOfPolynomials{0, 1, 0}(r, theta, phi) ==
        approx(y));
  CHECK(BallTestFunctions::ProductOfPolynomials{0, 0, 1}(r, theta, phi) ==
        approx(z));
  CHECK(BallTestFunctions::ProductOfPolynomials{2, 0, 0}(r, theta, phi) ==
        approx(x * x));
  CHECK(BallTestFunctions::ProductOfPolynomials{1, 1, 0}(r, theta, phi) ==
        approx(x * y));
  CHECK(BallTestFunctions::ProductOfPolynomials{1, 1, 1}(r, theta, phi) ==
        approx(x * y * z));
  CHECK(BallTestFunctions::ProductOfPolynomials{2, 2, 2}(r, theta, phi) ==
        approx(x * x * y * y * z * z));

  // DataVector overload gives the same result
  const DataVector r_v{r};
  const DataVector th_v{theta};
  const DataVector ph_v{phi};
  CHECK(BallTestFunctions::ProductOfPolynomials{2, 1, 0}(r_v, th_v, ph_v)[0] ==
        approx(x * x * y));
}

void test_derivatives() {
  // Generic point away from coordinate singularities
  const double r = 0.6;
  const double theta = 1.1;
  const double phi = 0.8;

  // Constant: all derivatives are exactly zero
  {
    const BallTestFunctions::ProductOfPolynomials f{0, 0, 0};
    const DataVector r_v{r};
    const DataVector th_v{theta};
    const DataVector ph_v{phi};
    CHECK(f.df_dr(r_v, th_v, ph_v)[0] == 0.0);
    CHECK(f.df_dth(r_v, th_v, ph_v)[0] == 0.0);
    CHECK(f.df_dph(r_v, th_v, ph_v)[0] == 0.0);
  }

  check_derivatives({1, 0, 0}, r, theta, phi);
  check_derivatives({0, 1, 0}, r, theta, phi);
  check_derivatives({0, 0, 1}, r, theta, phi);
  check_derivatives({2, 0, 0}, r, theta, phi);
  check_derivatives({0, 2, 0}, r, theta, phi);
  check_derivatives({0, 0, 2}, r, theta, phi);
  check_derivatives({1, 1, 0}, r, theta, phi);
  check_derivatives({1, 0, 1}, r, theta, phi);
  check_derivatives({2, 2, 0}, r, theta, phi);
  check_derivatives({2, 0, 2}, r, theta, phi);
}

void test_definite_integral() {
  const double pi = std::numbers::pi;

  // Odd powers integrate to zero
  CHECK(BallTestFunctions::ProductOfPolynomials{1, 0, 0}.definite_integral() ==
        0.0);
  CHECK(BallTestFunctions::ProductOfPolynomials{0, 1, 0}.definite_integral() ==
        0.0);
  CHECK(BallTestFunctions::ProductOfPolynomials{0, 0, 1}.definite_integral() ==
        0.0);
  CHECK(BallTestFunctions::ProductOfPolynomials{1, 2, 0}.definite_integral() ==
        0.0);

  // \int_{|r| \le 1} 1 dV = 4 \pi/ 3
  CHECK(BallTestFunctions::ProductOfPolynomials{0, 0, 0}.definite_integral() ==
        approx(4.0 * pi / 3.0));

  // \int_{|r| \le 1} x^2 dV = 4 \pi /15  (same for y^2 and z^2 by symmetry)
  CHECK(BallTestFunctions::ProductOfPolynomials{2, 0, 0}.definite_integral() ==
        approx(4.0 * pi / 15.0));
  CHECK(BallTestFunctions::ProductOfPolynomials{0, 2, 0}.definite_integral() ==
        approx(4.0 * pi / 15.0));
  CHECK(BallTestFunctions::ProductOfPolynomials{0, 0, 2}.definite_integral() ==
        approx(4.0 * pi / 15.0));

  // \int_{|r| \le 1} x^2 y^2 dV = 4 \pi /105
  CHECK(BallTestFunctions::ProductOfPolynomials{2, 2, 0}.definite_integral() ==
        approx(4.0 * pi / 105.0));
}

SPECTRE_TEST_CASE("TestHelpers.Numerical.Spectral.BallTestFunctions",
                  "[Unit][NumericalAlgorithms]") {
  test_function_values();
  test_derivatives();
  test_definite_integral();
}
}  // namespace
