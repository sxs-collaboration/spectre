// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <vector>  // IWYU pragma: keep

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Burgers/Linear.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/PointwiseFunctions/AnalyticSolutions/Burgers/CheckBurgersSolution.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_forward_declare Tensor

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Burgers.Linear",
                  "[PointwiseFunctions][Unit]") {
  const double shock_time = 1.5;
  const Burgers::Solutions::Linear solution(shock_time);
  const DataVector positions{-1., 0., 1., 2.5};
  check_burgers_solution(solution, positions, {-1., 0., 1., 2.5});

  // Check that the initial profile is correct.
  CHECK_ITERABLE_APPROX(
      get(solution.u(tnsr::I<DataVector, 1>{{{positions}}}, 0.)),
      positions / -shock_time);
  CHECK_ITERABLE_APPROX(get(serialize_and_deserialize(solution).u(
                            tnsr::I<DataVector, 1>{{{positions}}}, 0.)),
                        positions / -shock_time);

  const auto created_solution =
      test_creation<Burgers::Solutions::Linear>("  ShockTime: 1.5");
  const auto x = tnsr::I<DataVector, 1>{{{positions}}};
  const double t = 0.0;
  CHECK(created_solution.variables(x, t, tmpl::list<Burgers::Tags::U>{}) ==
        solution.variables(x, t, tmpl::list<Burgers::Tags::U>{}));
}
