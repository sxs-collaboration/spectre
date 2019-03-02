// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/VariableFixing/LimitLorentzFactor.hpp"
#include "PointwiseFunctions/Hydro/LorentzFactor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace VariableFixing {
namespace {
void test_variable_fixer(const LimitLorentzFactor& variable_fixer) {
  const Scalar<DataVector> density{DataVector{1.0, 2.0, 1.0e-5, 1.0e-6}};
  auto spatial_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(density, 1.0);
  get<0, 0>(spatial_metric) = 0.2;
  get<0, 1>(spatial_metric) = 0.3;
  get<0, 2>(spatial_metric) = 0.4;
  get<1, 1>(spatial_metric) = 0.5;
  get<1, 2>(spatial_metric) = 0.6;
  get<2, 2>(spatial_metric) = 0.7;

  tnsr::I<DataVector, 3, Frame::Inertial> spatial_velocity{
      {{DataVector{0.4999, 0.4999, 0.4999, 0.4999},
        DataVector{1.3567, 1.3566, 1.3567, 1.3566},
        DataVector{-0.200, -0.200, -0.200, -0.200}}}};

  auto lorentz_factor = hydro::lorentz_factor(
      dot_product(spatial_velocity, spatial_velocity, spatial_metric));
  const Scalar<DataVector> expected_lorentz_factor{
      DataVector{get(lorentz_factor)[0], get(lorentz_factor)[1], 50.0,
                 get(lorentz_factor)[3]}};
  const double rescale_velocity_factor =
      sqrt((1.0 - 1.0 / square(get(expected_lorentz_factor)[2])) /
           (1.0 - 1.0 / square(get(lorentz_factor)[2])));
  const tnsr::I<DataVector, 3, Frame::Inertial> expected_spatial_velocity{
      {{DataVector{0.4999, 0.4999, 0.4999 * rescale_velocity_factor, 0.4999},
        DataVector{1.3567, 1.3566, 1.3567 * rescale_velocity_factor, 1.3566},
        DataVector{-0.200, -0.200, -0.200 * rescale_velocity_factor, -0.200}}}};

  variable_fixer(make_not_null(&lorentz_factor),
                 make_not_null(&spatial_velocity), density);

  CHECK(lorentz_factor == expected_lorentz_factor);
  CHECK(spatial_velocity == expected_spatial_velocity);
  Approx custom_approx = Approx::custom().epsilon(6.0e-14).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(
      expected_lorentz_factor,
      hydro::lorentz_factor(
          dot_product(spatial_velocity, spatial_velocity, spatial_metric)),
      custom_approx);
}

SPECTRE_TEST_CASE("Unit.Evolution.VariableFixing.LimitLorentzFactor",
                  "[VariableFixing][Unit]") {
  SECTION("operator== and operator!=") {
    CHECK(LimitLorentzFactor{1.0, 8.0} == LimitLorentzFactor{1.0, 8.0});
    CHECK_FALSE(LimitLorentzFactor{2., 8.0} == LimitLorentzFactor{1.0, 8.0});
    CHECK_FALSE(LimitLorentzFactor{1.0, 9.0} == LimitLorentzFactor{1.0, 8.0});
    CHECK_FALSE(LimitLorentzFactor{1.0, 8.0} != LimitLorentzFactor{1.0, 8.0});
    CHECK(LimitLorentzFactor{2.0, 8.0} != LimitLorentzFactor{1.0, 8.0});
    CHECK(LimitLorentzFactor{1.0, 9.0} != LimitLorentzFactor{1.0, 8.0});
  }

  SECTION("variable fixing") {
    LimitLorentzFactor variable_fixer{1.0e-4, 50.0};
    test_variable_fixer(variable_fixer);
    test_serialization(variable_fixer);
    test_variable_fixer(serialize_and_deserialize(variable_fixer));

    const auto fixer_from_options = test_creation<LimitLorentzFactor>(
        "  MaxDensityCutoff: 1.0e-4\n"
        "  LorentzFactorCap: 50.0\n");
    test_variable_fixer(fixer_from_options);
  }
}
}  // namespace
}  // namespace VariableFixing
