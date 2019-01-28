// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "PointwiseFunctions/MathFunctions/Gaussian.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

template <size_t VolumeDim>
class MathFunction;

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.MathFunctions.Gaussian",
                  "[PointwiseFunctions][Unit]") {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> real_dis(-1, 1);
  std::uniform_real_distribution<> positive_dis(0, 1);

  // If the width is too small then the terms in the second derivative
  // can become very large and fail the test due to rounding errors.
  const double width = positive_dis(gen) + 0.5;
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
          approx((4. * square(point - center) / pow<4>(width) -
                  2. / square(width)) *
                 mapped_point));
    CHECK(gauss.third_deriv(point) ==
          approx(-4. * (point - center) *
                 (2. * square((point - center) / width) - 3.) / pow<4>(width) *
                 mapped_point));
  }

  const DataVector one{1., 1.};
  const auto mapped_point = amplitude * exp(-square((one - center) / width));
  const auto first_deriv = -2. * (one - center) / square(width) * mapped_point;
  const auto second_deriv =
      (4. * square(one - center) / pow<4>(width) - 2. / square(width)) *
      mapped_point;
  const auto third_deriv = -4. * (one - center) *
                           (2. * square((one - center) / width) - 3.) /
                           pow<4>(width) * mapped_point;
  CHECK_ITERABLE_APPROX(gauss(one), mapped_point);
  CHECK_ITERABLE_APPROX(gauss.first_deriv(one), first_deriv);
  CHECK_ITERABLE_APPROX(gauss.second_deriv(one), second_deriv);
  CHECK_ITERABLE_APPROX(gauss.third_deriv(one), third_deriv);

  test_serialization(gauss);
  test_serialization_via_base<MathFunction<1>, MathFunctions::Gaussian>(
      amplitude, width, center);
  // test operator !=
  const MathFunctions::Gaussian gauss2{amplitude, width, center - 1.0};
  CHECK(gauss != gauss2);
}

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.MathFunctions.Gaussian.Factory",
                  "[PointwiseFunctions][Unit]") {
  test_factory_creation<MathFunction<1>>(
      "  Gaussian:\n"
      "    Amplitude: 3\n"
      "    Width: 2\n"
      "    Center: -9");
  // Catch requires us to have at least one CHECK in each test
  // The Unit.PointwiseFunctions.MathFunctions.Gaussian.Factory does not need to
  // check anything
  CHECK(true);
}
