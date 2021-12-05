// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>
#include <tuple>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Elliptic/Systems/Xcts/Geometry.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/Xcts/VerifySolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/Tov.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Schwarzschild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/TovStar.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags/Conformal.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/RegisterDerivedWithCharm.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Xcts::Solutions {
namespace {

using TovCoordinates = RelativisticEuler::Solutions::TovCoordinates;

void test_solution(const TovCoordinates coord_system) {
  CAPTURE(coord_system);
  EquationsOfState::register_derived_with_charm();
  const auto created = TestHelpers::test_factory_creation<
      elliptic::analytic_data::AnalyticSolution, TovStar>(
      "TovStar:\n"
      "  CentralDensity: 1.e-3\n"
      "  EquationOfState:\n"
      "    PolytropicFluid:\n"
      "      PolytropicExponent: 2\n"
      "      PolytropicConstant: 1.\n"
      "  Coordinates: " +
      get_output(coord_system));
  REQUIRE(dynamic_cast<const TovStar*>(created.get()) != nullptr);
  const auto& solution = dynamic_cast<const TovStar&>(*created);
  {
    INFO("Properties");
    const auto& radial_solution = solution.radial_solution();
    CHECK(radial_solution.coordinate_system() == coord_system);
  }
  {
    INFO("Semantics");
    test_serialization(solution);
    test_copy_semantics(solution);
    auto move_solution = solution;
    test_move_semantics(std::move(move_solution), solution);
  }
  if (coord_system == TovCoordinates::Isotropic) {
    INFO("Test exterior solution is Schwarzschild");
    const auto& radial_solution = solution.radial_solution();
    const double star_radius = radial_solution.outer_radius();
    const double total_mass = radial_solution.total_mass();
    CAPTURE(star_radius);
    CAPTURE(total_mass);
    const Schwarzschild schwarzschild{total_mass,
                                      SchwarzschildCoordinates::Isotropic};
    // Look at a cube just outside the star
    const Mesh<3> mesh{5, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
    const auto logical_coords = logical_coordinates(mesh);
    const double dx = 0.1 * star_radius;
    using AffineMap = domain::CoordinateMaps::Affine;
    using AffineMap3D =
        domain::CoordinateMaps::ProductOf3Maps<AffineMap, AffineMap, AffineMap>;
    const domain::CoordinateMap<Frame::ElementLogical, Frame::Inertial,
                                AffineMap3D>
        coord_map{{{-1., 1., star_radius, star_radius + dx},
                   {-1., 1., 0., dx},
                   {-1., 1., 0., dx}}};
    const auto x = coord_map(logical_coords);
    const auto inv_jacobian = coord_map.inv_jacobian(logical_coords);
    // Check the TOV solution is identical to the Schwarzschild solution
    using all_fields = typename TovStar::template tags<DataVector>;
    const auto tov_vars = variables_from_tagged_tuple(
        solution.variables(x, mesh, inv_jacobian, all_fields{}));
    const auto schwarzschild_vars = variables_from_tagged_tuple(
        schwarzschild.variables(x, mesh, inv_jacobian, all_fields{}));
    CHECK_VARIABLES_APPROX(tov_vars, schwarzschild_vars);
  }
  {
    INFO("Verify the solution solves the XCTS equations");
    const auto& radial_solution = solution.radial_solution();
    const double star_radius = radial_solution.outer_radius();
    CAPTURE(star_radius);
    for (const double fraction_of_star_radius : {0., 0.5, 1., 1.5}) {
      CAPTURE(fraction_of_star_radius);
      const double inner_radius = fraction_of_star_radius * star_radius;
      const double outer_radius =
          (fraction_of_star_radius + 0.01) * star_radius;
      CAPTURE(inner_radius);
      CAPTURE(outer_radius);
      if (coord_system == TovCoordinates::Isotropic) {
        TestHelpers::Xcts::Solutions::verify_solution<
            Xcts::Geometry::FlatCartesian, 0>(
            solution, {{0., 0., 0.}}, inner_radius, outer_radius, 1.e-4);
      } else {
        TestHelpers::Xcts::Solutions::verify_solution<Xcts::Geometry::Curved,
                                                      0>(
            solution, {{0., 0., 0.}}, inner_radius, outer_radius, 1.e-4);
      }
    }
  }
}

// [[Timeout, 20]]
SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Xcts.TovStar",
                  "[PointwiseFunctions][Unit]") {
  test_solution(TovCoordinates::Schwarzschild);
  test_solution(TovCoordinates::Isotropic);
}

}  // namespace
}  // namespace Xcts::Solutions
