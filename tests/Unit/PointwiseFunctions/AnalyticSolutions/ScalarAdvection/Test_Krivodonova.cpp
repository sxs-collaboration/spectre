// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/AnalyticSolutions/ScalarAdvection/Krivodonova.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

void test_create() {
  const auto sol =
      TestHelpers::test_creation<ScalarAdvection::Solutions::Krivodonova>("");
  CHECK(sol == ScalarAdvection::Solutions::Krivodonova());
}

void test_serialize() {
  ScalarAdvection::Solutions::Krivodonova sol;
  test_serialization(sol);
}

void test_move() {
  ScalarAdvection::Solutions::Krivodonova sol;
  ScalarAdvection::Solutions::Krivodonova sol_copy;
  test_move_semantics(std::move(sol), sol_copy);  //  NOLINT
}

struct KrivodonovaProxy : ScalarAdvection::Solutions::Krivodonova {
  using ScalarAdvection::Solutions::Krivodonova::Krivodonova;

  using variables_tags = tmpl::list<ScalarAdvection::Tags::U>;

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<variables_tags> retrieve_variables(
      const tnsr::I<DataType, 1>& x, double t) const {
    return this->variables(x, t, variables_tags{});
  }
};

void verify_solution(const gsl::not_null<std::mt19937*> generator,
                     const size_t number_of_pts_per_subinterval) {
  // DataVectors to store values
  Scalar<DataVector> u_sol{9 * number_of_pts_per_subinterval};
  Scalar<DataVector> u_test{9 * number_of_pts_per_subinterval};

  // generate random 1D grid points within each of 9 subintervals.
  // - u = 0        : [-1.0, -0.8], [-0.6, -0.4], [-0.2, 0.0], [0.2, 0.4],
  //                  [0.6, 1.0]
  // - Gaussian     : [-0.8, -0.6]
  // - square pulse : [-0.4, -0.2]
  // - triangle     : [0.0, 0.2]
  // - half-ellipse : [0.4, 0.6]
  // Rather than generating random points for [-1.0, 1.0] without any rules,
  // patching the random grid points of these subintervals together allows for
  // this test to always 'hit' every on one of the shapes on the initial data,
  // which also helps for code coverage checks.
  std::uniform_real_distribution<> distribution_coords(-1.0, 1.0);
  auto x = make_with_random_values<tnsr::I<DataVector, 1>>(
      generator, make_not_null(&distribution_coords),
      9 * number_of_pts_per_subinterval);

  for (size_t i = 0; i < 9; ++i) {
    const auto x_sub = make_with_random_values<Scalar<DataVector>>(
        generator, make_not_null(&distribution_coords),
        number_of_pts_per_subinterval);
    if (i < 8) {
      // first 8 subintervals
      for (size_t j = 0; j < number_of_pts_per_subinterval; ++j) {
        get<0>(x)[i * number_of_pts_per_subinterval + j] =
            0.1 * get(x_sub)[j] - 0.9 + 0.2 * i;
      }
    } else {
      // handling the last subinterval [0.6, 1.0]
      for (size_t j = 0; j < number_of_pts_per_subinterval; ++j) {
        get<0>(x)[i * number_of_pts_per_subinterval + j] =
            0.2 * get(x_sub)[j] + 0.8;
      }
    }
  }

  // generate random time and shift coordinates
  std::uniform_real_distribution<> distribution_time(0.0, 10.0);
  double t{make_with_random_values<double>(generator,
                                           make_not_null(&distribution_time))};
  get<0>(x) = get<0>(x) + t;

  // test the solution
  KrivodonovaProxy sol;
  u_sol = get<ScalarAdvection::Tags::U>(sol.retrieve_variables(x, t));
  u_test = pypp::call<Scalar<DataVector>>("Krivodonova", "u", x, t);
  CHECK_ITERABLE_APPROX(u_sol, u_test);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.ScalarAdvection.Krivodonova",
    "[Unit][PointwiseFunctions]") {
  test_create();
  test_serialize();
  test_move();

  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/ScalarAdvection"};
  MAKE_GENERATOR(gen);
  verify_solution(make_not_null(&gen), 5);
}
