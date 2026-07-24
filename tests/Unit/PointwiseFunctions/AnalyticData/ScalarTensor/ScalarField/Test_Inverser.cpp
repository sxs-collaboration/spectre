// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "PointwiseFunctions/AnalyticData/ScalarTensor/ScalarField/Inverser.hpp"
#include "Utilities/TMPL.hpp"

namespace {

template <size_t Dim>
Scalar<DataVector> construct_profile_and_evaluate_scalar_field(
    const tnsr::I<DataVector, Dim, Frame::Inertial>& x, double amplitude) {
  const ScalarTensor::AnalyticData::ScalarField::Inverser<Dim>
      scalar_initial_guess(amplitude);
  return get<::CurvedScalarWave::Tags::Psi>(scalar_initial_guess.variables(
      x, tmpl::list<::CurvedScalarWave::Tags::Psi>{}));
}

template <size_t Dim>
tnsr::i<DataVector, Dim, Frame::Inertial>
construct_profile_and_evaluate_scalar_field_derivative(
    const tnsr::I<DataVector, Dim, Frame::Inertial>& x, double amplitude) {
  const ScalarTensor::AnalyticData::ScalarField::Inverser<Dim>
      scalar_initial_guess(amplitude);
  return get<::CurvedScalarWave::Tags::Phi<Dim, Frame::Inertial>>(
      scalar_initial_guess.variables(
          x,
          tmpl::list<::CurvedScalarWave::Tags::Phi<Dim, Frame::Inertial>>{}));
}
template <size_t Dim, typename DataType>
void test_inverser(const DataType& used_for_size) {
  auto f_scalar_field = &construct_profile_and_evaluate_scalar_field<Dim>;
  pypp::check_with_random_values<2>(f_scalar_field, "Inverser",
                                    "inverser_scalar_field",
                                    {{{1e-5, 1.}, {-1., 1.}}}, used_for_size);

  auto f_scalar_field_derivative =
      &construct_profile_and_evaluate_scalar_field_derivative<Dim>;
  pypp::check_with_random_values<2>(f_scalar_field_derivative, "Inverser",
                                    "inverser_scalar_field_derivative",
                                    {{{1e-5, 1.}, {-1., 1.}}}, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.ST.ScalarField.Inverser",
    "[Unit][PointwiseFunctions]") {
  const pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticData/ScalarTensor/ScalarField"};

  GENERATE_UNINITIALIZED_DATAVECTOR;

  test_inverser<3>(dv);
}
