// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <cmath>

#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/MathFunctions/PowX.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.MathFunctions.PowX",
                  "[PointwiseFunctions][Unit]") {
  const std::array<double, 3> test_values{{-1.4, 2.5, 0.}};

  // Check i = 0 and i = 1 cases seperately
  {
    MathFunctions::PowX power(0);
    for (size_t j = 0; j < 3; ++j) {
      const auto value = gsl::at(test_values, j);
      CHECK(power(value) == 1.);
      CHECK(power.first_deriv(value) == 0.);
      CHECK(power.second_deriv(value) == 0.);
    }
  }

  {
    MathFunctions::PowX power(1);
    for (size_t j = 0; j < 3; ++j) {
      const auto value = gsl::at(test_values, j);
      CHECK(power(value) == approx(value));
      CHECK(power.first_deriv(value) == 1.);
      CHECK(power.second_deriv(value) == 0.);
    }
  }

  // Check several more powers
  for (int i = 2; i < 5; ++i) {
    MathFunctions::PowX power(i);
    for (size_t j = 0; j < 3; ++j) {
      const auto value = gsl::at(test_values, j);
      CHECK(power(value) == approx(std::pow(value, i)));
      CHECK(power.first_deriv(value) == approx(i * std::pow(value, i - 1)));
      CHECK(power.second_deriv(value) ==
            approx(i * (i - 1) * std::pow(value, i - 2)));
    }
  }
  // Check negative powers
  for (int i = -5; i < -2; ++i) {
    MathFunctions::PowX power(i);
    // Don't check x=0 with a negative power
    for (size_t j = 0; j < 2; ++j) {
      const auto value = gsl::at(test_values, j);
      CHECK(power(value) == approx(std::pow(value, i)));
      CHECK(power.first_deriv(value) == approx(i * std::pow(value, i - 1)));
      CHECK(power.second_deriv(value) ==
            approx(i * (i - 1) * std::pow(value, i - 2)));
    }
  }

  MathFunctions::PowX power(3);
  const DataVector test_dv{2, 1.4};
  CHECK(power(test_dv) == pow(test_dv, 3));
  CHECK(power.first_deriv(test_dv) == 3 * pow(test_dv, 2));
  CHECK(power.second_deriv(test_dv) == 6 * test_dv);

  test_serialization(power);
  test_serialization_via_base<MathFunction<1>, MathFunctions::PowX>(3);
  // test operator !=
  const MathFunctions::PowX power2{-3};
  CHECK(power != power2);
}

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.MathFunctions.PowX.Factory",
                  "[PointwiseFunctions][Unit]") {
  test_factory_creation<MathFunction<1>>("  PowX:\n    Power: 3");
}
