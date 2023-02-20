// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Elliptic/Systems/Xcts/Geometry.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/Xcts/VerifySolution.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/RotatingStar.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/WrappedGr.hpp"

namespace Xcts::Solutions {

// [[Timeout, 20]]
SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Xcts.RotatingStar",
                  "[PointwiseFunctions][Unit]") {
  using RotatingStar = WrappedGrMhd<RelativisticEuler::Solutions::RotatingStar>;
  const std::string rotns_filename = unit_test_src_path() +
                                     "/PointwiseFunctions/AnalyticSolutions/"
                                     "RelativisticEuler/RotatingStarId.dat";
  const auto created = TestHelpers::test_factory_creation<
      elliptic::analytic_data::AnalyticSolution, RotatingStar>(
      "RotatingStar:\n"
      "  RotNsFilename: " +
      rotns_filename +
      "\n"
      "  PolytropicConstant: 100.\n");
  REQUIRE(dynamic_cast<const RotatingStar*>(created.get()) != nullptr);
  const auto& solution = dynamic_cast<const RotatingStar&>(*created);
  {
    INFO("Semantics");
    test_serialization(solution);
    test_copy_semantics(solution);
    auto move_solution = solution;
    test_move_semantics(std::move(move_solution), solution);
  }
  {
    INFO("Verify the solution solves the XCTS equations");
    const auto& equatorial_radius = solution.gr_solution().equatorial_radius();
    CAPTURE(equatorial_radius);
    for (const double fraction_of_star_radius : {0.2, 1., 1.5}) {
      CAPTURE(fraction_of_star_radius);
      const double inner_radius = fraction_of_star_radius * equatorial_radius;
      const double outer_radius =
          (fraction_of_star_radius + 0.01) * equatorial_radius;
      CAPTURE(inner_radius);
      CAPTURE(outer_radius);
      TestHelpers::Xcts::Solutions::verify_solution<Xcts::Geometry::Curved, 0>(
          solution, {{0., 0., 0.}}, inner_radius, outer_radius, 2.e-3);
    }
  }
}

}  // namespace Xcts::Solutions
