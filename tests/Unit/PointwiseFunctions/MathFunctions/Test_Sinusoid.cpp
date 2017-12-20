// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cmath>
#include <random>

#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/MathFunctions/Sinusoid.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.MathFunctions.Sinusoid",
                  "[PointwiseFunctions][Unit]") {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> real_dis(-1, 1);
  const double wavenumber = real_dis(gen);
  CAPTURE_PRECISE(wavenumber);
  const double amplitude = real_dis(gen);
  CAPTURE_PRECISE(amplitude);
  const double phase = real_dis(gen);
  CAPTURE_PRECISE(phase);
  const MathFunctions::Sinusoid sinusoid{amplitude, wavenumber, phase};

  // Check some random points
  for (size_t i = 0; i < 100; ++i) {
    const double point = real_dis(gen);
    const double mapped_point = amplitude * sin(wavenumber * point + phase);
    CHECK(sinusoid(point) == approx(mapped_point));
    CHECK(sinusoid.first_deriv(point) ==
          approx(amplitude * wavenumber * cos(wavenumber * point + phase)));
    CHECK(sinusoid.second_deriv(point) ==
          approx(-square(wavenumber) * mapped_point));
  }

  const DataVector one{1., 1.};
  const DataVector mapped_point = amplitude * sin(wavenumber * one + phase);
  const DataVector mapped_first_deriv =
      amplitude * wavenumber * cos(wavenumber * one + phase);
  const DataVector mapped_second_deriv = -square(wavenumber) * mapped_point;
  CHECK_ITERABLE_APPROX(sinusoid(one), mapped_point);
  CHECK_ITERABLE_APPROX(sinusoid.first_deriv(one), mapped_first_deriv);
  CHECK_ITERABLE_APPROX(sinusoid.second_deriv(one), mapped_second_deriv);

  test_serialization(sinusoid);
  test_serialization_via_base<MathFunction<1>, MathFunctions::Sinusoid>(
      amplitude, wavenumber, phase);
  // test operator !=
  const MathFunctions::Sinusoid sinusoid2{amplitude + 1.0, wavenumber, phase};
  CHECK(sinusoid != sinusoid2);
}

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.MathFunctions.Sinusoid.Factory",
                  "[PointwiseFunctions][Unit]") {
  test_factory_creation<MathFunction<1>>(
      "  Sinusoid:\n"
      "    Amplitude: 3\n"
      "    Wavenumber: 2\n"
      "    Phase: -9");
}
