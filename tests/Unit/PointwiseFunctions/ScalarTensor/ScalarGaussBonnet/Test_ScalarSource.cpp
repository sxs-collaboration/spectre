// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "PointwiseFunctions/ScalarTensor/ScalarGaussBonnet/ScalarSource.hpp"
#include "Utilities/Gsl.hpp"

namespace {
namespace detail {
Scalar<DataVector> test_gauss_bonnet_scalar_source(
    const Scalar<DataVector>& weyl_electric_scalar,
    const Scalar<DataVector>& weyl_magnetic_scalar,
    const Scalar<DataVector>& psi,
    const std::array<double, 3>& coupling_parameters, const double mass_psi,
    const double start_time, const double ramp_time, const double time) {
  const ScalarTensor::CouplingParameterOptions coupling_parameters_opts{
      gsl::at(coupling_parameters, 0), gsl::at(coupling_parameters, 1),
      gsl::at(coupling_parameters, 2)};
  return ::ScalarTensor::gauss_bonnet_scalar_source(
      weyl_electric_scalar, weyl_magnetic_scalar, psi, coupling_parameters_opts,
      mass_psi, std::pair<double, double>{start_time, ramp_time}, time);
}

Scalar<DataVector> test_multiply_by_negative_deriv_of_coupling_func(
    const Scalar<DataVector>& psi,
    const std::array<double, 3>& coupling_parameters, const double start_time,
    const double ramp_time, const double time) {
  Scalar<DataVector> result = make_with_value<Scalar<DataVector>>(psi, 1.0);
  const ScalarTensor::CouplingParameterOptions coupling_parameters_opts{
      gsl::at(coupling_parameters, 0), gsl::at(coupling_parameters, 1),
      gsl::at(coupling_parameters, 2)};
  ::ScalarTensor::multiply_by_negative_deriv_of_coupling_func(
      make_not_null(&result), psi, coupling_parameters_opts,
      std::pair<double, double>{start_time, ramp_time}, time);
  return result;
}

Scalar<DataVector> test_multiply_by_negative_second_deriv_of_coupling_func(
    const Scalar<DataVector>& psi,
    const std::array<double, 3>& coupling_parameters, const double start_time,
    const double ramp_time, const double time) {
  Scalar<DataVector> result = make_with_value<Scalar<DataVector>>(psi, 1.0);
  const ScalarTensor::CouplingParameterOptions coupling_parameters_opts{
      gsl::at(coupling_parameters, 0), gsl::at(coupling_parameters, 1),
      gsl::at(coupling_parameters, 2)};
  ::ScalarTensor::multiply_by_negative_second_deriv_of_coupling_func(
      make_not_null(&result), psi, coupling_parameters_opts,
      std::pair<double, double>{start_time, ramp_time}, time);
  return result;
}

}  // namespace detail

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.ScalarTensor.Sgb.ScalarSource",
                  "[Unit][PointwiseFunctions]") {
  const pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/ScalarTensor"};

  pypp::check_with_random_values<
      1,
      Scalar<DataVector> (*)(
          const Scalar<DataVector>&, const Scalar<DataVector>&,
          const Scalar<DataVector>&, const std::array<double, 3>&, const double,
          const double, const double, const double),
      DataVector, nullptr>(&detail::test_gauss_bonnet_scalar_source, "Sources",
                           {"gauss_bonnet_scalar_source"}, {{{1.0e-2, 0.5}}},
                           DataVector{5});

  pypp::check_with_random_values<1,
                                 Scalar<DataVector> (*)(
                                     const Scalar<DataVector>&,
                                     const std::array<double, 3>&, const double,
                                     const double, const double),
                                 DataVector, nullptr>(
      &detail::test_multiply_by_negative_deriv_of_coupling_func, "Sources",
      {"negative_deriv_of_coupling_func"}, {{{1.0e-2, 0.5}}}, DataVector{5});

  pypp::check_with_random_values<1,
                                 Scalar<DataVector> (*)(
                                     const Scalar<DataVector>&,
                                     const std::array<double, 3>&, const double,
                                     const double, const double),
                                 DataVector, nullptr>(
      &detail::test_multiply_by_negative_second_deriv_of_coupling_func,
      "Sources", {"negative_second_deriv_of_coupling_func"}, {{{1.0e-2, 0.5}}},
      DataVector{5});

}
}  // namespace
