// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>
#include <tuple>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Kerr.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Xcts::Solutions {
namespace {

void test_solution(const double mass, const std::array<double, 3> spin,
                   const std::array<double, 3> center,
                   const std::string& options_string) {
  CAPTURE(mass);
  CAPTURE(spin);
  CAPTURE(center);
  const auto created =
      TestHelpers::test_factory_creation<Xcts::Solutions::AnalyticSolution<
          tmpl::list<Xcts::Solutions::Registrars::Kerr>>>(options_string);
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
    // Once the XCTS equations are implemented we will check here that the
    // solution numerically solves the equations. That's both a rigorous test
    // that the solution is correctly implemented, as well as a test of the XCTS
    // system implementation. Until then we just call into the solution and
    // probe a few variables.
    const Mesh<3> mesh{12, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
    const size_t num_points = mesh.number_of_grid_points();
    const double inner_radius = 2. * mass;
    const double outer_radius = 5. * mass;
    using AffineMap = domain::CoordinateMaps::Affine;
    using AffineMap3D =
        domain::CoordinateMaps::ProductOf3Maps<AffineMap, AffineMap, AffineMap>;
    const domain::CoordinateMap<Frame::Logical, Frame::Inertial, AffineMap3D>
        coord_map{
            {{-1., 1., inner_radius + center[0], outer_radius + center[0]},
             {-1., 1., inner_radius + center[1], outer_radius + center[1]},
             {-1., 1., inner_radius + center[2], outer_radius + center[2]}}};
    const auto logical_coords = logical_coordinates(mesh);
    const auto inertial_coords = coord_map(logical_coords);
    const auto inv_jacobian = coord_map.inv_jacobian(logical_coords);
    const auto vars = solution.variables(
        inertial_coords, mesh, inv_jacobian,
        tmpl::list<Tags::ConformalFactor<DataVector>,
                   Tags::ShiftBackground<DataVector, 3, Frame::Inertial>>{});
    CHECK_ITERABLE_APPROX(get(get<Tags::ConformalFactor<DataVector>>(vars)),
                          DataVector(num_points, 1.));
    for (size_t i = 0; i < 3; ++i) {
      CHECK_ITERABLE_APPROX(
          SINGLE_ARG(
              get<Tags::ShiftBackground<DataVector, 3, Frame::Inertial>>(vars)
                  .get(i)),
          DataVector(num_points, 0.));
    }
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Xcts.Kerr",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/Xcts"};
  test_solution(0.43, {{0.1, 0.2, 0.3}}, {{1., 2., 3.}},
                "Kerr:\n"
                "  Mass: 0.43\n"
                "  Spin: [0.1, 0.2, 0.3]\n"
                "  Center: [1., 2., 3.]");
}

}  // namespace Xcts::Solutions
