// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <string>
#include <utility>

#include "Elliptic/Systems/Xcts/Geometry.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/Xcts/VerifySolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/HarmonicSchwarzschild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/WrappedGr.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"

namespace Xcts::Solutions {
namespace {

using HarmonicSchwarzschild = WrappedGr<gr::Solutions::HarmonicSchwarzschild>;

void test_solution(const double mass, const std::array<double, 3>& center,
                   const std::string& options_string) {
  CAPTURE(mass);
  CAPTURE(center);
  const auto created = TestHelpers::test_factory_creation<
      elliptic::analytic_data::AnalyticSolution, HarmonicSchwarzschild>(
      options_string);
  REQUIRE(dynamic_cast<const HarmonicSchwarzschild*>(created.get()) != nullptr);
  const auto& solution = dynamic_cast<const HarmonicSchwarzschild&>(*created);
  {
    INFO("Properties");
    CHECK(solution.gr_solution().mass() == mass);
    CHECK(solution.gr_solution().center() == center);
  }
  {
    INFO("Semantics");
    test_serialization(solution);
    test_copy_semantics(solution);
    auto move_solution = solution;
    test_move_semantics(std::move(move_solution), solution);
  }
  {
    INFO("Verify the solution solves the XCTS system");
    TestHelpers::Xcts::Solutions::verify_solution<Xcts::Geometry::Curved, 0>(
        solution, center, 2. * mass, 6. * mass, 1.e-5);
  }
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.Xcts.HarmonicSchwarzschild",
    "[PointwiseFunctions][Unit]") {
  test_solution(0.43, {{1., 2., 3.}},
                "HarmonicSchwarzschild:\n"
                "  Mass: 0.43\n"
                "  Center: [1., 2., 3.]");
}

}  // namespace Xcts::Solutions
