// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <pup.h>

#include "Domain/Creators/Brick.hpp"
#include "Domain/Domain.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/GrMhd/VerifyGrMhdSolution.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/RotatingStar.hpp"

namespace {
void verify_solution(const RelativisticEuler::Solutions::RotatingStar& solution,
                     const std::array<double, 3>& x,
                     const double error_tolerance) {
  CAPTURE(x);
  const std::array<double, 3> dx{{1.e-4, 1.e-4, 1.e-4}};
  domain::creators::Brick brick(x - dx, x + dx, {{0, 0, 0}}, {{5, 5, 5}},
                                {{false, false, false}});
  Mesh<3> mesh{brick.initial_extents()[0], Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto};
  const auto domain = brick.create_domain();

  verify_grmhd_solution(solution, domain.blocks()[0], mesh, error_tolerance,
                        1.234, 1.e-4);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.RelEuler.RotatingStar",
    "[Unit][PointwiseFunctions]") {
  PUPable_reg(RelativisticEuler::Solutions::RotatingStar);
  // You can uncomment the cst_solution code below, include <iostream>,
  // and update the path to the `dat` file in order to figure out what the
  // central rest mass density is.
  // RelativisticEuler::Solutions::detail::CstSolution cst_solution(
  //     "/path/to/InitialData.dat", 100);
  // std::cout << cst_solution.interpolate(0.0, 0.0, true) << "\n\n";

  // The tolerance depends on the resolution that the RotNS file used. In order
  // to avoid storing very large files in the repo, we accept a larger
  // tolerance. However, Nils Deppe verified on Jan. 5, 2022 that increasing the
  // RotNS code resolution allows for a stricter tolerance.
  const double error_tolerance = 4.e-6;
  const RelativisticEuler::Solutions::RotatingStar solution(
      dynamic_cast<const RelativisticEuler::Solutions::RotatingStar&>(
          *serialize_and_deserialize(
              RelativisticEuler::Solutions::RotatingStar(
                  unit_test_src_path() +
                      "/PointwiseFunctions/AnalyticSolutions/RelativisticEuler/"
                      "RotatingStarId.dat",
                  100)
                  .get_clone())));

  CHECK(solution.equatorial_radius() == approx(7.155891353887));

  // Near the center the finite difference derivatives cause "large" errors
  // (1e-8) and so the check fails.
  verify_solution(solution, {{1., 0., 0.}}, error_tolerance);
  verify_solution(solution, {{-1., 0., 0.}}, error_tolerance);
  verify_solution(solution, {{0., 1., 0.}}, error_tolerance);
  verify_solution(solution, {{0., -1., 0.}}, error_tolerance);
  verify_solution(solution, {{0., 0., 1.}}, error_tolerance);
  verify_solution(solution, {{0., 0., -1.}}, error_tolerance);
  verify_solution(solution, {{1., 0.528, 0.2}}, error_tolerance);
  verify_solution(solution, {{1., 0.528, -0.2}}, error_tolerance);
  verify_solution(solution, {{1., -0.528, 0.2}}, error_tolerance);
  verify_solution(solution, {{-1., 0.528, 0.2}}, error_tolerance);
  verify_solution(solution, {{1., -0.528, -0.2}}, error_tolerance);
  verify_solution(solution, {{-1., 0.528, -0.2}}, error_tolerance);
  verify_solution(solution, {{-1., -0.528, 0.2}}, error_tolerance);
  verify_solution(solution, {{-1., -0.528, -0.2}}, error_tolerance);
  // Test outside the star
  verify_solution(solution, {{20., 20., 20.}}, error_tolerance);
  verify_solution(solution, {{20., 20., -20.}}, error_tolerance);
  verify_solution(solution, {{20., -20., 20.}}, error_tolerance);
  verify_solution(solution, {{-20., 20., 20.}}, error_tolerance);
  verify_solution(solution, {{20., -20., -20.}}, error_tolerance);
  verify_solution(solution, {{-20., 20., -20.}}, error_tolerance);
  verify_solution(solution, {{-20., -20., 20.}}, error_tolerance);
  verify_solution(solution, {{-20., -20., -20.}}, error_tolerance);
  verify_solution(solution, {{20., 0., 0.}}, error_tolerance);
  verify_solution(solution, {{0., 20., 0.}}, error_tolerance);
  verify_solution(solution, {{0., 0., 20.}}, error_tolerance);
}
