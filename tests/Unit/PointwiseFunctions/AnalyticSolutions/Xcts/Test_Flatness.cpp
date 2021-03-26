// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <tuple>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Flatness.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Xcts::Solutions {

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Xcts.Flatness",
                  "[PointwiseFunctions][Unit]") {
  const auto created =
      TestHelpers::test_factory_creation<Xcts::Solutions::AnalyticSolution<
          tmpl::list<Xcts::Solutions::Registrars::Flatness>>>("Flatness");
  REQUIRE(dynamic_cast<const Flatness<>*>(created.get()) != nullptr);
  const auto& solution = dynamic_cast<const Flatness<>&>(*created);
  {
    INFO("Semantics");
    test_serialization(solution);
    test_copy_semantics(solution);
    auto move_solution = solution;
    test_move_semantics(std::move(move_solution), solution);
  }
  // Testing that each tag is either 1 or 0 is not so useful, since it just
  // duplicates the logic that is being tested. Instead, this test should verify
  // the solution solves the XCTS equations once those are implemented. Until
  // then, just check a selection of variables to test the solution runs without
  // error.
  const size_t num_points = 3;
  const tnsr::I<DataVector, 3> x{num_points,
                                 std::numeric_limits<double>::signaling_NaN()};
  const auto vars = solution.variables(
      x, tmpl::list<Tags::ConformalFactor<DataVector>,
                    Tags::ShiftExcess<DataVector, 3, Frame::Inertial>,
                    Tags::ConformalMetric<DataVector, 3, Frame::Inertial>>{});
  CHECK(get(get<Tags::ConformalFactor<DataVector>>(vars)) ==
        DataVector(num_points, 1.));
  const auto& shift_excess =
      get<Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>(vars);
  for (size_t i = 0; i < 3; ++i) {
    CHECK(shift_excess.get(i) == DataVector(num_points, 0.));
  }
  const auto& conformal_metric =
      get<Tags::ConformalMetric<DataVector, 3, Frame::Inertial>>(vars);
  for (size_t i = 0; i < 3; ++i) {
    CHECK(conformal_metric.get(i, i) == DataVector(num_points, 1.));
    for (size_t j = 0; j < i; ++j) {
      CHECK(conformal_metric.get(i, j) == DataVector(num_points, 0.));
    }
  }
}

}  // namespace Xcts::Solutions
