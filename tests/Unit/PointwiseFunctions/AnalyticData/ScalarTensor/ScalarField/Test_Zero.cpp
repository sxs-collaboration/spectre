// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/AnalyticData/ScalarTensor/ScalarField/Zero.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace {

template <size_t Dim>
void test_zero(const DataVector& used_for_size) {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-5., 5.);
  auto x = make_with_random_values<tnsr::I<DataVector, Dim, Frame::Inertial>>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  const DataVector zero = make_with_value<DataVector>(x, 0.);

  const ScalarTensor::AnalyticData::ScalarField::Zero<Dim> scalar_initial_guess;
  const Scalar<DataVector> scalar_field =
      get<::CurvedScalarWave::Tags::Psi>(scalar_initial_guess.variables(
          x, tmpl::list<CurvedScalarWave::Tags::Psi>{}));
  const tnsr::i<DataVector, Dim, Frame::Inertial> scalar_field_derivative =
      get<CurvedScalarWave::Tags::Phi<Dim, Frame::Inertial>>(
          scalar_initial_guess.variables(
              x,
              tmpl::list<CurvedScalarWave::Tags::Phi<Dim, Frame::Inertial>>{}));

  CHECK(get(scalar_field) == zero);
  for (size_t i = 0; i < Dim; ++i) {
    CHECK(scalar_field_derivative.get(i) == zero);
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticData.ST.ScalarField.Zero",
                  "[Unit][PointwiseFunctions]") {
  GENERATE_UNINITIALIZED_DATAVECTOR;
  test_zero<3>(dv);
}
