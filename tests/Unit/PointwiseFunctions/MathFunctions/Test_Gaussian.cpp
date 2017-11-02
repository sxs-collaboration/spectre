// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cmath>

#include "PointwiseFunctions/MathFunctions/Gaussian.hpp"
#include "tests/Unit/PointwiseFunctions/MathFunctions/TestMathHelpers.hpp"
#include "tests/Unit/TestFactoryCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.MathFunctions.Gaussian",
                  "[PointwiseFunctions][Unit]") {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> real_dis(-1, 1);
  std::uniform_real_distribution<> positive_dis(0, 1);

  const double width = positive_dis(gen);
  CAPTURE_PRECISE(width);
  const double amplitude = positive_dis(gen);
  CAPTURE_PRECISE(amplitude);
  const double center = real_dis(gen);
  CAPTURE_PRECISE(center);
  const MathFunctions::Gaussian gauss{amplitude, width, center};

  // Check some random points
  for (size_t i = 0; i < 100; ++i) {
    const double point = real_dis(gen);
    const double mapped_point =
        amplitude * exp(-square((point - center) / width));
    CHECK(gauss(point) == approx(mapped_point));
    CHECK(gauss.first_deriv(point) ==
          approx(-2. * (point - center) / square(width) * mapped_point));
    CHECK(gauss.second_deriv(point) ==
          approx(
              (4 * square(point - center) / pow<4>(width) - 2 / square(width)) *
              mapped_point));
  }

  const DataVector one{1., 1.};
  for (size_t s = 0; s < one.size(); ++s) {
    const auto mapped_point = amplitude * exp(-square((one - center) / width));
    CHECK(gauss(one)[s] == approx(mapped_point[s]));
    CHECK(gauss.first_deriv(one)[s] ==
          approx(-2. * (one[s] - center) / square(width) * mapped_point[s]));
    CHECK(gauss.second_deriv(one)[s] ==
          approx((4 * square(one[s] - center) / pow<4>(width) -
                  2 / square(width)) *
                 mapped_point[s]));
  }
  test_pup_function(gauss);
}

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.MathFunctions.Gaussian.Factory",
                  "[PointwiseFunctions][Unit]") {
  test_factory_creation<MathFunction<1>>(
      "  Gaussian:\n"
      "    Amplitude: 3\n"
      "    Width: 2\n"
      "    Center: -9");
}
