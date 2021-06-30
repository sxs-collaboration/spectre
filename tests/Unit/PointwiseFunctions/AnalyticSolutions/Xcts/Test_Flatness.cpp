// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <utility>

#include "Elliptic/Systems/Xcts/Geometry.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/Xcts/VerifySolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Flatness.hpp"

namespace Xcts::Solutions {

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Xcts.Flatness",
                  "[PointwiseFunctions][Unit]") {
  const auto created = TestHelpers::test_creation<
      std::unique_ptr<Xcts::Solutions::AnalyticSolution<
          tmpl::list<Xcts::Solutions::Registrars::Flatness>>>>("Flatness");
  REQUIRE(dynamic_cast<const Flatness<>*>(created.get()) != nullptr);
  const auto& solution = dynamic_cast<const Flatness<>&>(*created);
  {
    INFO("Semantics");
    test_serialization(solution);
    test_copy_semantics(solution);
    auto move_solution = solution;
    test_move_semantics(std::move(move_solution), solution);
  }
  {
    INFO("Verify the solution solves the XCTS equations");
    TestHelpers::Xcts::Solutions::verify_solution<Xcts::Geometry::FlatCartesian,
                                                  0>(solution, {{0., 0., 0.}},
                                                     0., 1., 1.e-13);
  }
}

}  // namespace Xcts::Solutions
