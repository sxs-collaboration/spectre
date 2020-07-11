// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
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
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Tov.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/TovStar.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace {

void test_create_from_options() noexcept {
  const auto star = TestHelpers::test_creation<
      RelativisticEuler::Solutions::TovStar<gr::Solutions::TovSolution>>(
      "CentralDensity: 1.0e-5\n"
      "PolytropicConstant: 0.001\n"
      "PolytropicExponent: 1.4");
  CHECK(star ==
        RelativisticEuler::Solutions::TovStar<gr::Solutions::TovSolution>(
            0.00001, 0.001, 1.4));
}

void test_move() noexcept {
  RelativisticEuler::Solutions::TovStar<gr::Solutions::TovSolution> star(
      1.e-4, 4.0, 2.5);
  RelativisticEuler::Solutions::TovStar<gr::Solutions::TovSolution> star_copy(
      1.e-4, 4.0, 2.5);
  test_move_semantics(std::move(star), star_copy);  //  NOLINT
}

void test_serialize() noexcept {
  RelativisticEuler::Solutions::TovStar<gr::Solutions::TovSolution> star(
      1.e-3, 8.0, 2.0);
  test_serialization(star);
}

void verify_solution(const RelativisticEuler::Solutions::TovStar<
                         gr::Solutions::TovSolution>& solution,
                     const std::array<double, 3>& x) noexcept {
  const std::array<double, 3> dx{{1.e-4, 1.e-4, 1.e-4}};
  domain::creators::Brick brick(x - dx, x + dx, {{false, false, false}},
                                {{0, 0, 0}}, {{5, 5, 5}});
  Mesh<3> mesh{brick.initial_extents()[0], Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto};
  const auto domain = brick.create_domain();
  verify_grmhd_solution(solution, domain.blocks()[0], mesh, 1.e-10, 1.234,
                        1.e-4);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.RelEuler.Tov",
                  "[Unit][PointwiseFunctions]") {
  test_create_from_options();
  test_serialize();
  test_move();

  RelativisticEuler::Solutions::TovStar<gr::Solutions::TovSolution> solution(
      1.0e-3, 100.0, 2.0);
  const double radius_of_star = 10.04735006833273303;
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
