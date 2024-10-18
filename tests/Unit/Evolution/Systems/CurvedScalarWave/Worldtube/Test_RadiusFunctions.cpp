// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>

#include "Evolution/Systems/CurvedScalarWave/Worldtube/RadiusFunctions.hpp"
#include "Framework/TestHelpers.hpp"

namespace CurvedScalarWave::Worldtube {
namespace {
void test_smooth_broken_power_law_limits() {
  const double amp = 1.234;
  const double rb = 43.45;
  for (const double exp : {0., 0.01, 0.5, 1., 1.5, 2.456, 3., 4.}) {
    for (const double delta : {0.01, 0.0345, 0.1}) {
      CAPTURE(exp, delta);
      const double r_left_1 = 1.1;
      const double r_left_2 = 1.2;
      const double v1 = smooth_broken_power_law(r_left_1, exp, amp, rb, delta);
      const double v2 = smooth_broken_power_law(r_left_2, exp, amp, rb, delta);
      const double left_exponent = log(v2 / v1) / log(r_left_2 / r_left_1);
      CHECK(left_exponent == approx(exp));

      const double r_right_1 = 100. * rb;
      const double r_right_2 = 100. * rb + 0.1;
      const double v1_const =
          smooth_broken_power_law(r_right_1, exp, amp, rb, delta);
      const double v2_const =
          smooth_broken_power_law(r_right_2, exp, amp, rb, delta);
      CHECK(v1_const == approx(v2_const));
    }
  }
}

double central_difference_order8(const std::function<double(double)>& func,
                                 const double x, const double h) {
  return (3. * func(x - 4. * h) - 32. * func(x - 3. * h) +
          168. * func(x - 2. * h) - 672. * func(x - h) + 672. * func(x + h) +
          -168. * func(x + 2. * h) + 32. * func(x + 3. * h) -
          3. * func(x + 4. * h)) /
         (840. * h);
}

void test_smooth_broken_power_law_derivative() {
  const double amp = 1.234;
  const double rb = 10.;

  // the finite difference method carries an error of about 1e-10
  const Approx custom_approx = Approx::custom().epsilon(1.e-10).scale(1.0);

  for (const double exp : {0., 0.01, 0.5, 1., 1.5, 2.456, 3., 4.}) {
    for (const double delta : {0.01, 0.0345, 0.1}) {
      auto fd_func = [exp, amp, rb, delta](double r) -> double {
        return smooth_broken_power_law(r, exp, amp, rb, delta);
      };
      // test for a wide range of radii: critical are the points around rb in
      // the transition region and around rb + 1000 * delta, where the
      // derivative automatically returns 0.
      for (const double test_radius : {0.1, 5., rb - delta, rb, rb + delta, 15.,
                                       rb + 999. * delta, rb + 1001. * delta}) {
        CAPTURE(exp, delta, test_radius);
        const double fd_derivative =
            central_difference_order8(fd_func, test_radius, 1e-3);
        const double analytical_derivative = smooth_broken_power_law_derivative(
            test_radius, exp, amp, rb, delta);
        CHECK(fd_derivative == custom_approx(analytical_derivative));
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.CurvedScalarWave.RadiusFunctions",
                  "[Unit][Evolution]") {
  test_smooth_broken_power_law_limits();
  test_smooth_broken_power_law_derivative();
}
}  // namespace
}  // namespace CurvedScalarWave::Worldtube
