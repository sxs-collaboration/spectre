// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/FixConservatives.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {

template <size_t Dim>
void test_fix_conservatives(
    const RelativisticEuler::Valencia::FixConservatives<Dim>& variable_fixer) {
  Scalar<DataVector> tilde_d{DataVector{1.2, 9.e-13, 2.0, 0.0, 8.7}};
  Scalar<DataVector> tilde_tau{DataVector{-0.001, 0.7, 0.1, 5.5, -0.05}};
  auto tilde_s =
      make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(tilde_d, 0.0);
  for (size_t i = 0; i < Dim; ++i) {
    tilde_s.get(i) = DataVector{1.25, -1.56, 4.32, 5.0, 0.0};
  }
  // Use identity matrix as metric for simplicity
  auto sqrt_det_spatial_metric =
      make_with_value<Scalar<DataVector>>(tilde_d, 1.0);
  auto inv_spatial_metric =
      make_with_value<tnsr::II<DataVector, Dim, Frame::Inertial>>(tilde_d, 0.0);
  for (size_t i = 0; i < Dim; ++i) {
    inv_spatial_metric.get(i, i) = get(sqrt_det_spatial_metric);
  }
  auto tilde_s_squared = make_with_value<Scalar<DataVector>>(tilde_d, 0.0);
  dot_product(make_not_null(&tilde_s_squared), tilde_s, tilde_s,
              inv_spatial_metric);

  // Second and fourth elements must be set to D_min = 1.e-12
  Scalar<DataVector> expected_tilde_d_after_fixing{
      DataVector{1.2, 1.e-12, 2.0, 1.e-12, 8.7}};

  // First and fifth elements must be set to zero
  Scalar<DataVector> expected_tilde_tau_after_fixing{
      DataVector{0.0, 0.7, 0.1, 5.5, 0.0}};

  auto expected_tilde_s_after_fixing = tilde_s;
  const double safety_factor_for_s_tilde = 1.e-11;
  for (size_t s = 0; s < get(tilde_d).size(); ++s) {
    const double tilde_s_squared_max =
        get(expected_tilde_tau_after_fixing)[s] *
        (get(expected_tilde_tau_after_fixing)[s] +
         2.0 * get(expected_tilde_d_after_fixing)[s]);
    if (get(tilde_s_squared)[s] >
        (1 - safety_factor_for_s_tilde) * tilde_s_squared_max) {
      for (size_t i = 0; i < Dim; ++i) {
        expected_tilde_s_after_fixing.get(i)[s] *= std::min(
            1.0,
            sqrt((1 - safety_factor_for_s_tilde) * tilde_s_squared_max /
                 (get(tilde_s_squared)[s] + 1.e-16 * square(get(tilde_d)[s]))));
      }
    }
  }

  variable_fixer(&tilde_d, &tilde_tau, &tilde_s, inv_spatial_metric,
                 sqrt_det_spatial_metric);
  CHECK_ITERABLE_APPROX(tilde_d, expected_tilde_d_after_fixing);
  CHECK_ITERABLE_APPROX(tilde_tau, expected_tilde_tau_after_fixing);
  CHECK_ITERABLE_APPROX(tilde_s, expected_tilde_s_after_fixing);

  // Trying to fix the already fixed values shouldn't change anything.
  variable_fixer(&tilde_d, &tilde_tau, &tilde_s, inv_spatial_metric,
                 sqrt_det_spatial_metric);
  CHECK_ITERABLE_APPROX(tilde_d, expected_tilde_d_after_fixing);
  CHECK_ITERABLE_APPROX(tilde_tau, expected_tilde_tau_after_fixing);
  CHECK_ITERABLE_APPROX(tilde_s, expected_tilde_s_after_fixing);

  test_serialization(variable_fixer);

  const auto fixer_from_options =
      test_creation<RelativisticEuler::Valencia::FixConservatives<Dim>>(
          "  MinimumValueOfD: 1.0e-12\n"
          "  CutoffD: 1.2e-12\n"
          "  SafetyFactorForS: 1.0e-11");
  CHECK(variable_fixer == fixer_from_options);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.RelativisticEuler.Valencia.FixConservatives",
                  "[Unit][RelativisticEuler]") {
  test_fix_conservatives(RelativisticEuler::Valencia::FixConservatives<1>{
      1.e-12, 1.2e-12, 1.0e-11});
  test_fix_conservatives(RelativisticEuler::Valencia::FixConservatives<2>{
      1.e-12, 1.2e-12, 1.0e-11});
  test_fix_conservatives(RelativisticEuler::Valencia::FixConservatives<3>{
      1.e-12, 1.2e-12, 1.0e-11});
}
