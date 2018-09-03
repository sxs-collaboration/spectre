// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cmath>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/VariableFixing/RadiallyFallingFloor.hpp"
#include "Utilities/Gsl.hpp"

namespace grmhd {
namespace Tags {
struct Pressure;
struct RestMassDensity;
}  // namespace Tags
}  // namespace grmhd

SPECTRE_TEST_CASE("Unit.Evolution.VariableFixing.RadiallyFallingFloor",
                  "[VariableFixing][Unit]") {
  Scalar<DataVector> broken_pressure{DataVector{0.0, 1.e-8, 2.0, -5.5, 3.2}};
  Scalar<DataVector> broken_density{DataVector{2.3, -4.2, 1.e-10, 0.0, -0.1}};
  const double root_three = sqrt(3.0);
  constexpr double one_third = 1.0 / 3.0;
  const DataVector fixed_pressure{
      1.e-7 * pow(2.0 * root_three, -2.5) * one_third, 1.e-8, 2.0,
      1.e-7 * pow(3, -1.25) * one_third, 3.2};
  const DataVector fixed_density{
      2.3, 1.e-5 * pow(3, -0.75),
      1.e-10,  // quantities at a radius below
               // `radius_at_which_to_begin_applying_floor` do not get fixed.
      1.e-5 * pow(3, -0.75), 1.e-5 * pow(2.0 * root_three, -1.5)};
  const DataVector x{-2.0, -1.0, 0.0, 1.0, 2.0};
  const DataVector y{-2.0, -1.0, 0.0, 1.0, 2.0};
  const DataVector z{-2.0, -1.0, 0.0, 1.0, 2.0};
  tnsr::I<DataVector, 3, Frame::Inertial> coords{{{x, y, z}}};
  const double radius_at_which_to_begin_applying_floor = 1.e-4;
  VariableFixing::RadiallyFallingFloor<
      3, grmhd::Tags::RestMassDensity,
      grmhd::Tags::Pressure>::apply(&broken_density, &broken_pressure, coords,
                                    radius_at_which_to_begin_applying_floor);
  CHECK_ITERABLE_APPROX(broken_pressure.get(), fixed_pressure);
  CHECK_ITERABLE_APPROX(broken_density.get(), fixed_density);
}
