// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "PointwiseFunctions/ScalarTensor/ScalarGaussBonnet/CouplingFunctions/Exponential.hpp"
#include "PointwiseFunctions/ScalarTensor/ScalarGaussBonnet/CouplingFunctions/QuarticPolynomial.hpp"

namespace {

template <typename CouplingType, typename... CouplingParametersTypes>
Scalar<DataVector> construct_couplingfunction_and_evaluate_coupling(
    const Scalar<DataVector>& scalar_field,
    CouplingParametersTypes... parameters) {
  CouplingType coupling(parameters...);
  Scalar<DataVector> result{};
  coupling.coupling_function(&result, scalar_field);
  CHECK(coupling.coupling_function(scalar_field) == result);
  return result;
}

template <typename CouplingType, typename... CouplingParametersTypes>
Scalar<DataVector> construct_couplingfunction_and_evaluate_coupling_prime(
    const Scalar<DataVector>& scalar_field,
    CouplingParametersTypes... parameters) {
  CouplingType coupling(parameters...);
  Scalar<DataVector> result{};
  coupling.coupling_function_prime(&result, scalar_field);
  CHECK(coupling.coupling_function_prime(scalar_field) == result);
  return result;
}

template <typename CouplingType, typename... CouplingParametersTypes>
Scalar<DataVector> construct_couplingfunction_and_evaluate_coupling_prime_prime(
    const Scalar<DataVector>& scalar_field,
    CouplingParametersTypes... parameters) {
  CouplingType coupling(parameters...);
  Scalar<DataVector> result{};
  coupling.coupling_function_prime_prime(&result, scalar_field);
  CHECK(coupling.coupling_function_prime_prime(scalar_field) == result);
  return result;
}

template <typename DataType>
void test_exponential_coupling(const DataType& used_for_size) {
  auto f = &construct_couplingfunction_and_evaluate_coupling<
      ScalarTensor::sgb::CouplingFunctions::Exponential, double, double>;
  pypp::check_with_random_values<1>(f, "CouplingFunctions",
                                    "exponential_coupling", {{{-1., 1.}}},
                                    used_for_size);

  auto f_prime = &construct_couplingfunction_and_evaluate_coupling_prime<
      ScalarTensor::sgb::CouplingFunctions::Exponential, double, double>;
  pypp::check_with_random_values<1>(f_prime, "CouplingFunctions",
                                    "exponential_coupling_prime", {{{-1., 1.}}},
                                    used_for_size);

  auto f_prime_prime =
      &construct_couplingfunction_and_evaluate_coupling_prime_prime<
          ScalarTensor::sgb::CouplingFunctions::Exponential, double, double>;
  pypp::check_with_random_values<1>(f_prime_prime, "CouplingFunctions",
                                    "exponential_coupling_prime_prime",
                                    {{{-1., 1.}}}, used_for_size);
}

template <typename DataType>
void test_quarticpolynomial_coupling(const DataType& used_for_size) {
  auto f = &construct_couplingfunction_and_evaluate_coupling<
      ScalarTensor::sgb::CouplingFunctions::QuarticPolynomial, double, double,
      double, double>;
  pypp::check_with_random_values<1>(f, "CouplingFunctions",
                                    "quarticpolynomial_coupling", {{{-1., 1.}}},
                                    used_for_size);

  auto f_prime = &construct_couplingfunction_and_evaluate_coupling_prime<
      ScalarTensor::sgb::CouplingFunctions::QuarticPolynomial, double, double,
      double, double>;
  pypp::check_with_random_values<1>(f_prime, "CouplingFunctions",
                                    "quarticpolynomial_coupling_prime",
                                    {{{-1., 1.}}}, used_for_size);

  auto f_prime_prime =
      &construct_couplingfunction_and_evaluate_coupling_prime_prime<
          ScalarTensor::sgb::CouplingFunctions::QuarticPolynomial, double,
          double, double, double>;
  pypp::check_with_random_values<1>(f_prime_prime, "CouplingFunctions",
                                    "quarticpolynomial_coupling_prime_prime",
                                    {{{-1., 1.}}}, used_for_size);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.ScalarTensor.sgb.CouplingFunctions",
                  "[Unit][PointwiseFunctions]") {
  const pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/ScalarTensor/ScalarGaussBonnet/CouplingFunctions"};

  GENERATE_UNINITIALIZED_DATAVECTOR;

  test_exponential_coupling(dv);
  test_quarticpolynomial_coupling(dv);
}
