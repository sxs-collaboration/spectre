// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <climits>
#include <cstddef>
#include <string>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/DerivZ4Constraint.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
template <size_t Dim, typename DataType>
void test_compute_grad_spatial_z4_constraint(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::ij<DataType, Dim, Frame::Inertial> (*)(
          const tnsr::i<DataType, Dim, Frame::Inertial>&,
          const tnsr::ii<DataType, Dim, Frame::Inertial>&,
          const tnsr::Ijj<DataType, Dim, Frame::Inertial>&,
          const tnsr::ijj<DataType, Dim, Frame::Inertial>&,
          const tnsr::I<DataType, Dim, Frame::Inertial>&,
          const tnsr::iJ<DataType, Dim, Frame::Inertial>&)>(
          &Ccz4::grad_spatial_z4_constraint<Dim, Frame::Inertial, DataType>),
      "DerivZ4Constraint", "grad_spatial_z4_constraint", {{{-1., 1.}}},
      used_for_size);
}

// Test that when \f$\hat{\Gamma}^i == \tilde{\Gamma}^i\f$ and \f$Z_i == 0\f$,
// the gradient of \f$Z_i\f$ is 0
template <typename Generator, typename DataType>
void test_grad_spatial_z4_vanishes(const gsl::not_null<Generator*> generator,
                                   const DataType& used_for_size) {
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  // \f$Z_i == 0\f$
  const auto spatial_z4_constraint =
      make_with_value<tnsr::i<DataType, 3, Frame::Inertial>>(used_for_size,
                                                             0.0);
  const auto conformal_spatial_metric =
      make_with_random_values<tnsr::ii<DataType, 3, Frame::Inertial>>(
          generator, distribution, used_for_size);
  const auto christoffel_second_kind =
      make_with_random_values<tnsr::Ijj<DataType, 3, Frame::Inertial>>(
          generator, distribution, used_for_size);
  const auto field_d =
      make_with_random_values<tnsr::ijj<DataType, 3, Frame::Inertial>>(
          generator, distribution, used_for_size);
  // \f$\hat{\Gamma}^i == \tilde{\Gamma}^i\f$
  const auto gamma_hat_minus_contracted_conformal_christoffel =
      make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(used_for_size,
                                                             0.0);
  // \f$\hat{\Gamma}^i == \tilde{\Gamma}^i\f$
  const auto d_gamma_hat_minus_contracted_conformal_christoffel =
      make_with_value<tnsr::iJ<DataType, 3, Frame::Inertial>>(used_for_size,
                                                              0.0);

  using result_tensor_type = tnsr::ij<DataType, 3, Frame::Inertial>;
  const result_tensor_type expected_grad_z4_constraint =
      make_with_value<result_tensor_type>(used_for_size, 0.0);

  const result_tensor_type actual_grad_z4_constraint =
      Ccz4::grad_spatial_z4_constraint(
          spatial_z4_constraint, conformal_spatial_metric,
          christoffel_second_kind, field_d,
          gamma_hat_minus_contracted_conformal_christoffel,
          d_gamma_hat_minus_contracted_conformal_christoffel);

  CHECK_ITERABLE_APPROX(actual_grad_z4_constraint, expected_grad_z4_constraint);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.DerivZ4Constraint",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env("Evolution/Systems/Ccz4/");

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_compute_grad_spatial_z4_constraint,
                                    (1, 2, 3));

  MAKE_GENERATOR(generator);
  test_grad_spatial_z4_vanishes(make_not_null(&generator),
                                std::numeric_limits<double>::signaling_NaN());
  test_grad_spatial_z4_vanishes(
      make_not_null(&generator),
      DataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
