// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cmath>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/VariableFixing/RadiallyFallingFloor.hpp"
#include "Utilities/Gsl.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {

void test_variable_fixer(
    const VariableFixing::RadiallyFallingFloor<3>& variable_fixer) {
  Scalar<DataVector> pressure{DataVector{0.0, 1.e-8, 2.0, -5.5, 3.2}};
  Scalar<DataVector> density{DataVector{2.3, -4.2, 1.e-10, 0.0, -0.1}};
  const double root_three = sqrt(3.0);
  constexpr double one_third = 1.0 / 3.0;
  const DataVector expected_pressure{
      1.e-7 * pow(2.0 * root_three, -2.5) * one_third, 1.e-8, 2.0,
      1.e-7 * pow(3, -1.25) * one_third, 3.2};
  const DataVector expected_density{
      2.3, 1.e-5 * pow(3, -0.75),
      1.e-10,  // quantities at a radius below
               // `radius_at_which_to_begin_applying_floor` do not get fixed.
      1.e-5 * pow(3, -0.75), 1.e-5 * pow(2.0 * root_three, -1.5)};
  const DataVector x{-2.0, -1.0, 0.0, 1.0, 2.0};
  const DataVector y{-2.0, -1.0, 0.0, 1.0, 2.0};
  const DataVector z{-2.0, -1.0, 0.0, 1.0, 2.0};
  tnsr::I<DataVector, 3, Frame::Inertial> coords{{{x, y, z}}};
  variable_fixer(&density, &pressure, coords);
  CHECK_ITERABLE_APPROX(pressure.get(), expected_pressure);
  CHECK_ITERABLE_APPROX(density.get(), expected_density);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.VariableFixing.RadiallyFallingFloor",
                  "[VariableFixing][Unit]") {
  VariableFixing::RadiallyFallingFloor<3> variable_fixer{1.e-4, 1.e-5, -1.5,
                                                         1.e-7 / 3.0, -2.5};
  test_variable_fixer(variable_fixer);
  test_serialization(variable_fixer);

  const auto fixer_from_options =
      test_creation<VariableFixing::RadiallyFallingFloor<3>>(
          "  MinimumRadius: 1.e-4\n"
          "  ScaleDensityFloor: 1.e-5\n"
          "  PowerDensityFloor: -1.5\n"
          "  ScalePressureFloor: 0.33333333333333333e-7\n"
          "  PowerPressureFloor: -2.5\n");
  test_variable_fixer(fixer_from_options);
}
