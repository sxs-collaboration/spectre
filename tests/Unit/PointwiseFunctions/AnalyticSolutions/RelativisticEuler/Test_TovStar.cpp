// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <random>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Domain.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/GrMhd/VerifyGrMhdSolution.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/TovStar.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace RelativisticEuler::Solutions {
namespace {

using TovCoordinates = gr::Solutions::TovCoordinates;

void verify_solution(const TovStar& solution, const std::array<double, 3>& x) {
  const std::array<double, 3> dx{{1.e-4, 1.e-4, 1.e-4}};
  domain::creators::Brick brick(x - dx, x + dx, {{0, 0, 0}}, {{5, 5, 5}},
                                {{false, false, false}});
  Mesh<3> mesh{brick.initial_extents()[0], Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto};
  const auto domain = brick.create_domain();
  verify_grmhd_solution(solution, domain.blocks()[0], mesh, 1.e-10, 1.234,
                        1.e-4);
}

void test_tov_star(const TovCoordinates coord_system) {
  Parallel::register_classes_with_charm<
      RelativisticEuler::Solutions::TovStar>();
  const std::unique_ptr<evolution::initial_data::InitialData> option_solution =
      TestHelpers::test_option_tag_factory_creation<
          evolution::initial_data::OptionTags::InitialData,
          RelativisticEuler::Solutions::TovStar>(
          "TovStar:\n"
          "  CentralDensity: 1.0e-3\n"
          "  PolytropicConstant: 100.0\n"
          "  PolytropicExponent: 2.0\n"
          "  Coordinates: " +
          get_output(coord_system));
  const auto deserialized_option_solution =
      serialize_and_deserialize(option_solution);
  const auto& solution =
      dynamic_cast<const RelativisticEuler::Solutions::TovStar&>(
          *deserialized_option_solution);
  {
    INFO("Semantics");
    CHECK(solution == TovStar{1.0e-3, 100.0, 2.0, coord_system});
    test_copy_semantics(solution);
    TovStar star_copy = solution;
    test_move_semantics(std::move(star_copy), solution);
    test_serialization(solution);
  }
  if (coord_system == TovCoordinates::Schwarzschild) {
    INFO("Properties");
    Approx custom_approx = Approx::custom().epsilon(1.0e-08).scale(1.0);
    CHECK(solution.radial_solution().outer_radius() ==
          custom_approx(10.0473500683));
    // Check a second solution
    TovStar second_solution{1.0e-3, 8.0, 2.0};
    CHECK(second_solution.radial_solution().outer_radius() ==
          custom_approx(3.4685521362));
  }

  // Check the solution numerically solves the GRMHD equations
  const double radius_of_star = solution.radial_solution().outer_radius();
  verify_solution(solution, {{0.0, 0.0, 0.0}});
  verify_solution(solution, {{0.0, 0.0, 0.5 * radius_of_star}});
  verify_solution(solution, {{0.0, radius_of_star, 0.0}});
  verify_solution(solution, {{1.5 * radius_of_star, 0.0, 0.0}});
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> real_dis(-1.1 * radius_of_star,
                                            1.1 * radius_of_star);
  std::array<double, 3> random_point{};
  for (size_t i = 0; i < 3; i++) {
    gsl::at(random_point, i) = real_dis(gen);
  }
  verify_solution(solution, random_point);
}

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.RelEuler.Tov",
                  "[Unit][PointwiseFunctions]") {
  test_tov_star(TovCoordinates::Schwarzschild);
  test_tov_star(TovCoordinates::Isotropic);
}

}  // namespace
}  // namespace RelativisticEuler::Solutions
