// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <vector>  // IWYU pragma: keep

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Burgers/Bump.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/PointwiseFunctions/AnalyticSolutions/Burgers/CheckBurgersSolution.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_forward_declare Tensor

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Burgers.Bump",
                  "[PointwiseFunctions][Unit]") {
  const double half_width = 5.;
  const double height = 3.;
  const double center = -8.;
  const Burgers::Solutions::Bump solution(half_width, height, center);
  const DataVector positions{-11., -9., -8., -6.};
  check_burgers_solution(solution, positions, {-0.5, 0., 0.1, 0.5});

  // Check that the initial profile is correct.
  CHECK_ITERABLE_APPROX(
      get(solution.u(tnsr::I<DataVector, 1>{{{positions}}}, 0.)),
      height * (1. - square((positions - center) / half_width)));
  CHECK_ITERABLE_APPROX(
      get(serialize_and_deserialize(solution).u(
          tnsr::I<DataVector, 1>{{{positions}}}, 0.)),
      height * (1. - square((positions - center) / half_width)));

  const auto created_solution = test_creation<Burgers::Solutions::Bump>(
      "  HalfWidth: 5.\n"
      "  Height: 3.\n"
      "  Center: -8.\n");
  const auto x = tnsr::I<DataVector, 1>{{{positions}}};
  const double t = 0.0;
  CHECK(created_solution.variables(x, t, tmpl::list<Burgers::Tags::U>{}) ==
        solution.variables(x, t, tmpl::list<Burgers::Tags::U>{}));
}
