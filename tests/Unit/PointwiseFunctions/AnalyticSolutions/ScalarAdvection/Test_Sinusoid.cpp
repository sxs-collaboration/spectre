// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "PointwiseFunctions/AnalyticData/ScalarAdvection/Sinusoid.hpp"
#include "PointwiseFunctions/AnalyticSolutions/ScalarAdvection/Sinusoid.hpp"

namespace {
void test_sinusoid_solution(const gsl::not_null<std::mt19937*> generator,
                            const size_t number_of_pts) {
  // DataVectors to store values
  Scalar<DataVector> u_sol{number_of_pts};
  Scalar<DataVector> u_test{number_of_pts};
  Scalar<DataVector> dudt_sol{number_of_pts};
  Scalar<DataVector> dudt_test{number_of_pts};

  // generate random 1D grid points
  std::uniform_real_distribution<> distribution(-1.0, 1.0);
  const auto x = make_with_random_values<tnsr::I<DataVector, 1>>(
      generator, make_not_null(&distribution), u_sol);

  // check if analytic solution equals to analytic data at t=0
  ScalarAdvection::AnalyticData::Sinusoid initial_data;
  ScalarAdvection::Solutions::Sinusoid solution;
  u_sol = solution.u<DataVector>(x, 0.0);
  u_test = initial_data.u<DataVector>(x);
  CHECK_ITERABLE_APPROX(u_sol, u_test);

  // compare u and du_dt values from analytic solution with python
  // implementation
  double t{0.3};
  u_sol = solution.u<DataVector>(x, t);
  u_test = pypp::call<Scalar<DataVector>>("Sinusoid", "u", x, t);
  dudt_sol = solution.du_dt<DataVector>(x, t);
  dudt_test = pypp::call<Scalar<DataVector>>("Sinusoid", "du_dt", x, t);
  CHECK_ITERABLE_APPROX(u_sol, u_test);
  CHECK_ITERABLE_APPROX(dudt_sol, dudt_test);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.ScalarAdvection.Sinusoid",
    "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/ScalarAdvection"};

  MAKE_GENERATOR(gen);
  test_sinusoid_solution(make_not_null(&gen), 10);
}
