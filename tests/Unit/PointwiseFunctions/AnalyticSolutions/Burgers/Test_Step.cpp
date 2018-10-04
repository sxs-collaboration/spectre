// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <vector>  // IWYU pragma: keep

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Burgers/Step.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/PointwiseFunctions/AnalyticSolutions/Burgers/CheckBurgersSolution.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_forward_declare Tensor

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Burgers.Step",
                  "[PointwiseFunctions][Unit]") {
  const double left_value = 2.3;
  const double right_value = 1.2;
  const double initial_position = -0.5;
  const Burgers::Solutions::Step solution(left_value, right_value,
                                          initial_position);
  const DataVector positions{-1., 0., 1., 2.5};

  // Check that the initial profile is correct.
  const DataVector expected_initial_profile =
      right_value +
      (left_value - right_value) *
          step_function(DataVector(-positions + initial_position));
  CHECK_ITERABLE_APPROX(
      get(solution.u(tnsr::I<DataVector, 1>{{{positions}}}, 0.)),
      expected_initial_profile);

  // Check serialization
  CHECK_ITERABLE_APPROX(get(serialize_and_deserialize(solution).u(
                            tnsr::I<DataVector, 1>{{{positions}}}, 0.)),
                        expected_initial_profile);

  // Check consistency
  // Note that we avoid checking at any times where x_i == x0 + speed * t,
  // because at these times the shock is near a grid point and the jump in the
  // solution breaks the tests of check_burgers_solution
  check_burgers_solution(solution, positions, {0., 0.7, 1.2, 10.0});

  const auto created_solution = test_creation<Burgers::Solutions::Step>(
      "  LeftValue: 2.3\n"
      "  RightValue: 1.2\n"
      "  InitialPosition: -0.5");
  const auto x = tnsr::I<DataVector, 1>{{{positions}}};
  const double t = 0.0;
  CHECK(created_solution.variables(x, t, tmpl::list<Burgers::Tags::U>{}) ==
        solution.variables(x, t, tmpl::list<Burgers::Tags::U>{}));
}
