// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "Evolution/Systems/ScalarAdvection/TimeDerivativeTerms.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {

template <size_t Dim>
void test_time_derivative(const gsl::not_null<std::mt19937*> generator,
                          const size_t number_of_pts) {
  // create datavectors to store computed values of dudt and flux
  Scalar<DataVector> dudt{number_of_pts, 0.0};
  tnsr::I<DataVector, Dim, Frame::Inertial> flux(number_of_pts);

  // generate random u and velocity field
  std::uniform_real_distribution<> distribution(-1.0, 1.0);
  const auto u = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&distribution), dudt);
  const auto velocity =
      make_with_random_values<tnsr::I<DataVector, Dim, Frame::Inertial>>(
          generator, make_not_null(&distribution), dudt);

  // evaluate dudt and flux
  ScalarAdvection::TimeDerivativeTerms<Dim>::apply(&dudt, &flux, u, velocity);

  // expected values of dudt and flux. note that dudt_expected is set to be
  // equal to the original value of dudt (which is 0.0 here), since we have no
  // source terms and the TimeDerivativeTerms::apply function should not make
  // any change on dudt argument.
  const Scalar<DataVector> dudt_expected{number_of_pts, 0.0};
  const tnsr::I<DataVector, Dim, Frame::Inertial> flux_expected{
      pypp::call<tnsr::I<DataVector, Dim, Frame::Inertial>>(
          "TestFunctions", "compute_flux", u, velocity)};

  // check values
  CHECK_ITERABLE_APPROX(dudt, dudt_expected);
  CHECK_ITERABLE_APPROX(flux, flux_expected);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ScalarAdvection.TimeDerivative",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/ScalarAdvection"};
  MAKE_GENERATOR(gen);

  test_time_derivative<1>(make_not_null(&gen), 2);
  test_time_derivative<2>(make_not_null(&gen), 5);
}
