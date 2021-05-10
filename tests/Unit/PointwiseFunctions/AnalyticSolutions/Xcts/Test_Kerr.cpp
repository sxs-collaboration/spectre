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
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Kerr.hpp"
#include "Utilities/TMPL.hpp"

namespace Xcts::Solutions {
namespace {

void test_solution(const double mass, const std::array<double, 3> spin,
                   const std::array<double, 3>& center,
                   const std::string& options_string) {
  CAPTURE(mass);
  CAPTURE(spin);
  CAPTURE(center);
  const auto created = TestHelpers::test_creation<
      std::unique_ptr<Xcts::Solutions::AnalyticSolution<
          tmpl::list<Xcts::Solutions::Registrars::Kerr>>>>(options_string);
  REQUIRE(dynamic_cast<const Kerr<>*>(created.get()) != nullptr);
  const auto& solution = dynamic_cast<const Kerr<>&>(*created);
  {
    INFO("Properties");
    CHECK(solution.mass() == mass);
    CHECK(solution.dimensionless_spin() == spin);
    CHECK(solution.center() == center);
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

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Xcts.Kerr",
                  "[PointwiseFunctions][Unit]") {
  test_solution(0.43, {{0.1, 0.2, 0.3}}, {{1., 2., 3.}},
                "Kerr:\n"
                "  Mass: 0.43\n"
                "  Spin: [0.1, 0.2, 0.3]\n"
                "  Center: [1., 2., 3.]");
}

}  // namespace Xcts::Solutions
