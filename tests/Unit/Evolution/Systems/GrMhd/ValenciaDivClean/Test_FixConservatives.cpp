// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservatives.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_include <array>

// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare VariableFixing::FixToAtmosphere

namespace {

void test_variable_fixer(
    const VariableFixing::FixConservatives& variable_fixer) {
  // Call variable fixer at four points
  // [0]:  tilde_d is too small, should be raised to limit
  // [1]:  tilde_tau is too small, raise to level of needed, which also
  //       causes tilde_s to be zeroed
  // [2]:  tilde_S is too big, so it is lowered
  // [3]:  all values are good, no changes

  Scalar<DataVector> tilde_d{DataVector{2.e-12, 1.0, 1.0, 1.0}};
  Scalar<DataVector> tilde_tau{DataVector{4.5, 1.5, 4.5, 4.5}};
  auto tilde_s =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(tilde_d, 0.0);
  auto tilde_b =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(tilde_d, 0.0);
  for (size_t d = 0; d < 3; ++d) {
    tilde_s.get(0) = DataVector{3.0, 0.0, 6.0, 5.0};
    tilde_b.get(1) = DataVector{2.0, 2.0, 2.0, 2.0};
  }

  auto expected_tilde_d = tilde_d;
  get(expected_tilde_d)[0] = 1.e-12;
  auto expected_tilde_tau = tilde_tau;
  get(expected_tilde_tau)[1] = 2.0;
  auto expected_tilde_s = tilde_s;
  expected_tilde_s.get(0)[2] = sqrt(27.0);

  auto spatial_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(tilde_d, 0.0);
  auto inv_spatial_metric =
      make_with_value<tnsr::II<DataVector, 3, Frame::Inertial>>(tilde_d, 0.0);
  auto sqrt_det_spatial_metric =
      make_with_value<Scalar<DataVector>>(tilde_d, 1.0);
  for (size_t d = 0; d < 3; ++d) {
    spatial_metric.get(d, d) = get(sqrt_det_spatial_metric);
    inv_spatial_metric.get(d, d) = get(sqrt_det_spatial_metric);
  }

  variable_fixer(&tilde_d, &tilde_tau, &tilde_s, tilde_b, spatial_metric,
                 inv_spatial_metric, sqrt_det_spatial_metric);

  CHECK_ITERABLE_APPROX(tilde_d, expected_tilde_d);
  CHECK_ITERABLE_APPROX(tilde_tau, expected_tilde_tau);
  CHECK_ITERABLE_APPROX(tilde_s, expected_tilde_s);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.GrMhd.ValenciaDivClean.FixConservatives",
                  "[VariableFixing][Unit]") {
  VariableFixing::FixConservatives variable_fixer{1.e-12, 1.0e-11, 0.0, 0.0};
  test_variable_fixer(variable_fixer);
  test_serialization(variable_fixer);

  const auto fixer_from_options =
      test_creation<VariableFixing::FixConservatives>(
          "  MinimumValueOfD: 1.0e-12\n"
          "  CutoffD: 1.0e-11\n"
          "  SafetyFactorForB: 0.0\n"
          "  SafetyFactorForS: 0.0\n");
  test_variable_fixer(fixer_from_options);
}
